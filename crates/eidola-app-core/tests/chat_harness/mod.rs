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
    /// 200 SSE stream that the server aborts mid-event (writes a partial event,
    /// then drops the TCP connection). Exercises the mid-SSE read failure arm.
    StreamingMidAbort,
    /// Non-2xx JSON error body (e.g. 500). Exercises the non-2xx arm of both
    /// `chat` and `chat_stream`.
    Non2xx(u16),
    /// Accept the request, then drop the connection before sending any
    /// response bytes (network error after send).
    DropBeforeResponse,
}

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
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            chat: ChatBehavior::OkBlocking,
            refund: RefundMode::Succeed,
            balance: 10_000_000,
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
    let refund_hits = Arc::new(AtomicU64::new(0));

    let task = {
        let issuer = issuer.clone();
        let chat_hits = chat_hits.clone();
        let refund_hits = refund_hits.clone();
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let issuer = issuer.clone();
                let config = config.clone();
                let chat_hits = chat_hits.clone();
                let refund_hits = refund_hits.clone();
                tokio::spawn(async move {
                    let _ = handle_conn(stream, issuer, config, chat_hits, refund_hits).await;
                });
            }
        })
    };

    MockServer {
        base_url,
        chat_hits,
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

async fn handle_conn(
    mut stream: TcpStream,
    issuer: Arc<Issuer>,
    config: MockConfig,
    chat_hits: Arc<AtomicU64>,
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
        ("GET", "/v1/models") => {
            write_json(&mut stream, 200, &models_body()).await?;
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
            refund_hits.fetch_add(1, Ordering::SeqCst);
            handle_refund(&mut stream, &issuer, &config, req.auth.as_deref()).await?;
        }
        ("POST", "/v1/chat/completions") => {
            chat_hits.fetch_add(1, Ordering::SeqCst);
            handle_chat(&mut stream, &issuer, &config, req.auth.as_deref()).await?;
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
            let mut body = serde_json::json!({
                "choices": [{ "message": { "role": "assistant", "content": "Hello from the mock." } }],
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
        ChatBehavior::OkStreaming => write_sse_stream(stream, true).await,
        ChatBehavior::StreamingMidAbort => write_sse_stream(stream, false).await,
    }
}

/// Write a chunked SSE stream. When `complete`, emits reasoning + content
/// deltas, a usage chunk, and `[DONE]`. When not complete, emits one partial
/// event and then drops the connection mid-stream (simulating an abort).
async fn write_sse_stream(stream: &mut TcpStream, complete: bool) -> std::io::Result<()> {
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
        "choices": [{ "delta": { "content": "Hello from the stream." } }]
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
    serde_json::json!({
        "data": [{
            "id": issuer.key_id_hex,
            "public_key": issuer.public_key_b64(),
            "domain_separator": format!("ACT-v1:{DS_ORG}:{DS_SERVICE}:{DS_DEPLOYMENT}:{DS_VERSION}"),
            "issue_from": "2026-01-01T00:00:00Z",
            "issue_until": "2030-01-01T00:00:00Z",
            "accept_until": "2030-01-01T00:00:00Z",
        }]
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
    let core = AppCore::with_test_http_client(config_dir, data_dir, client);
    let mock = core.runtime().block_on(async { start(config).await });
    core.set_base_url(mock.base_url.clone())
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
