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
    /// As `OkStreaming`, but the model mimics the per-message header
    /// scaffolding: its content starts with a header-shaped line. Exercises
    /// strip-on-receipt.
    OkStreamingWithHeader,
    /// 200 SSE stream that the server aborts mid-event (writes a partial event,
    /// then drops the TCP connection). Exercises the mid-SSE read failure arm.
    StreamingMidAbort,
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
}

/// The tool name the tool-calling behaviours call — matches
/// `eidola_app_core::tools::EchoTool`.
pub const TOOL_NAME: &str = "echo";

/// Whether the refund endpoints (inline + recovery) actually mint a successor
/// credential, or fail. Lets refund-recovery-succeeds vs -fails be selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefundMode {
    /// Refund endpoints return a valid successor credential.
    Succeed,
    /// `/v1/credentials/refund` returns 500; inline refunds are omitted.
    Fail,
}

/// Mock upstream configuration.
#[derive(Clone)]
pub struct MockConfig {
    pub chat: ChatBehavior,
    pub refund: RefundMode,
    /// Account available balance returned by `/v1/account/balances`.
    pub balance: i64,
    /// When set, `GET /v1/models` responds with this HTTP status instead of the
    /// pricing catalog — simulating a `prepare_turn` setup failure (the fetch
    /// the original PR #218 screenshot failed on) before the turn's own error
    /// wrapping exists.
    pub models_status: Option<u16>,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            chat: ChatBehavior::OkBlocking,
            refund: RefundMode::Succeed,
            balance: 10_000_000,
            models_status: None,
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
    _task: tokio::task::JoinHandle<()>,
}

impl MockServer {
    pub fn chat_hits(&self) -> u64 {
        self.chat_hits.load(Ordering::SeqCst)
    }
    pub fn refund_hits(&self) -> u64 {
        self.refund_hits.load(Ordering::SeqCst)
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

/// The shared human participant's label.
pub const HUMAN_LABEL: &str = "You";

/// The rendering-protocol note app-core appends to every turn's system
/// message. Pinned here verbatim (the same discipline as pinning the rendered
/// message bytes) — a change to `HEADER_PROTOCOL_NOTE` in `lib.rs` must be a
/// deliberate edit here too.
pub const HEADER_PROTOCOL_NOTE: &str = "Each message in this conversation begins with a one-line header identifying the post and \
     its author: `#<handle> · <author>`. Handles are assigned by the client; never write a \
     header line yourself — reply with your message text only.";

/// The thread-map protocol note app-core appends to the system message of a
/// turn whose space has branches (task 21). Pinned verbatim, same discipline as
/// [`HEADER_PROTOCOL_NOTE`].
pub const THREAD_MAP_NOTE: &str = "This space is threaded: the conversation above is one branch of \
     it, and other branches exist. A `<thread-map>` block appears as the last message listing \
     them — it is client-generated metadata, not a post by any participant, and no reply is due \
     to it.";

/// The exact `system` message content a turn sends: the responding
/// participant's effective system prompt (when it has one) followed by the
/// rendering-protocol note.
pub fn system_message(prompt: Option<&str>) -> String {
    system_message_with(prompt, &[])
}

/// [`system_message`] plus the extra notes a turn appends (the thread-map
/// notes), joined by blank lines exactly as `prepare_turn` does.
pub fn system_message_with(prompt: Option<&str>, extra_notes: &[&str]) -> String {
    let mut notes = vec![HEADER_PROTOCOL_NOTE];
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

/// The exact `<thread-map>` message content a turn appends.
///
/// `forks` is `(anchor_line, entry_lines)` — the expected bytes written out
/// long-hand, which is the point: this is a byte pin, not a reimplementation.
pub fn thread_map(forks: &[(String, Vec<String>)], respond_to: &str) -> String {
    let mut out = String::from(
        "<thread-map>\nBranches of this space that the conversation above does not contain. Each \
         line: handle · author · posts · last activity — opening line.\n",
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
    out.push_str(&format!("\nRespond to #{respond_to}.\n</thread-map>"));
    out
}

/// One thread-map entry line: `#handle · author · posts · when — opening`.
pub fn map_entry(item_id: &str, author: &str, posts: &str, when: &str, opening: &str) -> String {
    format!(
        "#{} · {author} · {posts} · {when} — {opening}",
        eidola_app_core::post_handle(item_id)
    )
}

/// The exact rendered upstream content of a post: its `#<handle> · <label>`
/// header line, a blank line, then the text.
pub fn headed(item_id: &str, label: &str, text: &str) -> String {
    let handle = eidola_app_core::post_handle(item_id);
    if text.is_empty() {
        format!("#{handle} · {label}")
    } else {
        format!("#{handle} · {label}\n\n{text}")
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

    let task = {
        let issuer = issuer.clone();
        let chat_hits = chat_hits.clone();
        let chat_bodies = chat_bodies.clone();
        let chat_auths = chat_auths.clone();
        let chat_auth_values = chat_auth_values.clone();
        let refund_hits = refund_hits.clone();
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
        ("GET", "/v1/models") => match config.models_status {
            Some(status) => {
                write_json(&mut stream, status, r#"{"error":"models unavailable"}"#).await?;
            }
            None => {
                write_json(&mut stream, 200, &models_body()).await?;
            }
        },
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
            refund_hits.fetch_add(1, Ordering::SeqCst);
            handle_refund(&mut stream, &issuer, &config, req.auth.as_deref()).await?;
        }
        ("POST", "/v1/chat/completions") => {
            // 1-based index of this chat request — the tool-calling
            // behaviours script their answer per round.
            let hit = chat_hits.fetch_add(1, Ordering::SeqCst) + 1;
            if let Ok(body) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                chat_bodies.lock().unwrap().push(body);
            }
            chat_auths.lock().unwrap().push(req.auth.is_some());
            chat_auth_values.lock().unwrap().push(req.auth.clone());
            handle_chat(&mut stream, &issuer, &config, req.auth.as_deref(), hit).await?;
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
) -> std::io::Result<()> {
    if config.refund == RefundMode::Fail {
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
) -> std::io::Result<()> {
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
        ChatBehavior::OkStreaming => write_sse_stream(stream, true, STREAM_CONTENT).await,
        ChatBehavior::OkStreamingWithHeader => {
            write_sse_stream(
                stream,
                true,
                &format!("{MIMICKED_HEADER}\n\n{STREAM_CONTENT}"),
            )
            .await
        }
        ChatBehavior::StreamingMidAbort => write_sse_stream(stream, false, STREAM_CONTENT).await,
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
                write_sse_stream(stream, true, TOOL_FINAL_CONTENT).await
            }
        }
        ChatBehavior::ToolRoundsStreamingWithExtras(rounds) => {
            if hit <= rounds {
                write_sse_tool_stream(stream, hit, true).await
            } else {
                write_sse_stream(stream, true, TOOL_FINAL_CONTENT).await
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

/// A header-shaped first line a model might mimic (see
/// `ChatBehavior::OkStreamingWithHeader`). Deliberately *not* a real handle of
/// anything — strip-on-receipt is shape-based, not identity-based.
pub const MIMICKED_HEADER: &str = "#a2c3d4e · Gemma4 31b";

/// Write a chunked SSE stream. When `complete`, emits reasoning + content
/// deltas, a usage chunk, and `[DONE]`. When not complete, emits one partial
/// event and then drops the connection mid-stream (simulating an abort).
async fn write_sse_stream(
    stream: &mut TcpStream,
    complete: bool,
    content_text: &str,
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

    let content = serde_json::json!({
        "choices": [{ "delta": { "content": content_text } }]
    });
    stream.write_all(&send_event(content.to_string())).await?;

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

fn models_body() -> String {
    serde_json::json!({
        "data": [{
            "id": MODEL,
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
