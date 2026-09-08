//! The proxy's HTTP surface: read a request, authenticate it, answer it.
//!
//! Deliberately **transport-agnostic** — it takes any async reader/writer, so
//! the process that owns the socket owns *binding* it while the answers stay
//! beside the [`AppCore`] methods they wrap. That is the same split the local
//! control protocol takes ([`crate::ipc`]), and it is what lets the whole
//! surface be exercised over an in-memory duplex with no socket, no port and
//! no privileges.
//!
//! ## What it exposes, and what it refuses to
//!
//! Three routes, and they are the whole stateless OpenAI surface:
//!
//! - `GET /v1/models` — what the exposed backends offer.
//! - `GET /v1/models/{id}` — one of them.
//! - `POST /v1/chat/completions` — inference, streaming or not.
//!
//! Everything else is a `404`, and that is a **capability** statement rather
//! than a gap, the same one [`crate::ipc`]'s verb surface makes: what has no
//! route cannot be reached. There is no conversation, space, participant,
//! template, wallet, account or record route here and there must never be one.
//! A downstream tool is a stranger to this profile — it authenticates with a
//! key the reader generated for it, not as the reader — so the line is drawn
//! at *stateless inference*, which is the whole of what the feature promised.
//!
//! ## Authentication
//!
//! `Authorization: Bearer <key>`, checked against the digests in `proxy_key`.
//! **A proxy with no live keys refuses everything** rather than running open:
//! "no keys yet" and "everyone welcome" must not be the same state, and the
//! first is what a freshly enabled proxy is.
//!
//! That header is consumed here and **never forwarded** — see
//! [`super::route::UpstreamHeaders`], which is the enumeration of what does go.

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited, StreamBody, combinators::BoxBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{AUTHORIZATION, CACHE_CONTROL, CONTENT_TYPE};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite};

use super::route::{ProxyChatRequest, ProxyStreamEvent};
use crate::AppCore;
use crate::error::AppError;
use crate::ipc::Shutdown;

/// The largest request body the proxy will read.
///
/// A conversation a tool replays can be genuinely large, so this is generous —
/// but it is a number rather than "whatever the peer sends", because the body
/// is buffered whole before it can be parsed and a local caller with a bug
/// should not be able to ask this process for unbounded memory.
const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// A response body, either complete or streaming.
type ProxyBody = BoxBody<Bytes, Infallible>;

/// Serve one connection until the peer goes away.
///
/// The latch is the socket owner's "the process is going away" signal, read at
/// the last moment before any work starts — because an abort lands only at an
/// await point, and the stretch from a request arriving to its dispatch passes
/// only awaits that are typically already ready. Exactly the seam
/// [`crate::ipc::Shutdown`] exists for, and the same latch type, because it is
/// the same problem: this surface can start **billed** work.
pub async fn serve_connection<I>(core: Arc<AppCore>, io: I, shutdown: Shutdown)
where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service = service_fn(move |request: Request<Incoming>| {
        let core = Arc::clone(&core);
        let shutdown = shutdown.clone();
        async move { Ok::<_, Infallible>(answer(core, request, shutdown).await) }
    });
    // HTTP/1.1 only. HTTP/2 without TLS needs prior knowledge, which no
    // OpenAI-compatible client sends, so supporting it would be surface with
    // no caller.
    let _ = hyper::server::conn::http1::Builder::new()
        .serve_connection(hyper_util::rt::TokioIo::new(io), service)
        .await;
}

/// Answer one request.
async fn answer(
    core: Arc<AppCore>,
    request: Request<Incoming>,
    shutdown: Shutdown,
) -> Response<ProxyBody> {
    if shutdown.is_latched() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "Eidola is shutting down",
            None,
        );
    }

    let presented = bearer_token(&request);
    let authenticated = match presented {
        Some(key) => core
            .authenticate_proxy_key(key.clone())
            .await
            .unwrap_or(false),
        None => false,
    };
    if !authenticated {
        // One answer for "no key", "wrong key" and "revoked key". Which of the
        // three it was is not something a caller holding the wrong one is owed,
        // and the reader can see every key's standing in Settings.
        return error_response(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "Invalid API key. Generate one in Eidola under Settings ▸ Proxy.",
            Some("invalid_api_key"),
        );
    }

    let path = request.uri().path().to_string();
    match (request.method(), path.as_str()) {
        (&Method::GET, "/v1/models") => models_response(&core, None).await,
        (&Method::GET, path) if path.starts_with("/v1/models/") => {
            let id = percent_decode(&path["/v1/models/".len()..]);
            models_response(&core, Some(id)).await
        }
        (&Method::POST, "/v1/chat/completions") => completions_response(core, request).await,
        _ => error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "Eidola's local proxy serves model listing and chat completions only.",
            None,
        ),
    }
}

/// The bearer token a request presents, if it presents one in that form.
fn bearer_token<B>(request: &Request<B>) -> Option<String> {
    let value = request.headers().get(AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

/// Minimal percent-decoding for a path segment — enough for a model id
/// carrying `@` or `/`, which is what this app's qualified ids and Hugging
/// Face-style names contain. Anything undecodable is left as it stands, which
/// simply fails to match a model and answers `404`.
fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).ok();
            if let Some(byte) = hex.and_then(|hex| u8::from_str_radix(hex, 16).ok()) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| segment.to_string())
}

/// `GET /v1/models`, and its single-model sibling.
async fn models_response(core: &AppCore, one: Option<String>) -> Response<ProxyBody> {
    let models = match core.proxy_models().await {
        Ok(models) => models,
        Err(e) => return app_error_response(&e),
    };
    match one {
        None => json_response(
            StatusCode::OK,
            json!({
                "object": "list",
                "data": models.iter().map(model_entry).collect::<Vec<Value>>(),
            }),
        ),
        Some(id) => match models.iter().find(|m| m.id == id) {
            Some(model) => json_response(StatusCode::OK, model_entry(model)),
            None => error_response(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                &format!("The model `{id}` does not exist or is not exposed by this proxy."),
                Some("model_not_found"),
            ),
        },
    }
}

/// One model, in the shape an OpenAI client reads.
///
/// `owned_by` names the Eidola backend the model is reached through, which is
/// the one piece of Eidola-specific information a tool genuinely needs: the
/// same model name can be served by an on-device engine for free and by the
/// hosted service for money, and the qualified id already says which — this
/// puts it somewhere a listing UI will actually show.
fn model_entry(model: &crate::ModelInfo) -> Value {
    json!({
        "id": model.id,
        "object": "model",
        "created": 0,
        "owned_by": backend_of(&model.id),
    })
}

fn backend_of(id: &str) -> String {
    crate::backends::parse_model_ref(id).backend_id
}

/// `POST /v1/chat/completions`.
async fn completions_response(
    core: Arc<AppCore>,
    request: Request<Incoming>,
) -> Response<ProxyBody> {
    let collected = match Limited::new(request.into_body(), MAX_REQUEST_BYTES)
        .collect()
        .await
    {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "The request body is larger than this proxy will read.",
                None,
            );
        }
    };
    let body: Value = match serde_json::from_slice(&collected) {
        Ok(body) => body,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("The request body is not valid JSON: {e}"),
                None,
            );
        }
    };
    let parsed = match ProxyChatRequest::from_json(&body) {
        Ok(parsed) => parsed,
        Err(e) => return app_error_response(&e),
    };

    if !parsed.stream {
        return match core.proxy_chat(parsed).await {
            Ok(answer) => json_response(
                StatusCode::from_u16(answer.status).unwrap_or(StatusCode::OK),
                answer.body,
            ),
            Err(e) => app_error_response(&e),
        };
    }

    // Streaming. The head cannot be written until the upstream has accepted,
    // or a failure would arrive as a `200` with an empty body — so the turn is
    // started and the first event decides. `ProxyStreamEvent::Open` is that
    // decision, sent once and only after the upstream's status is known.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ProxyStreamEvent>();
    let streaming = tokio::spawn(async move { core.proxy_chat_stream(parsed, tx).await });
    match rx.recv().await {
        Some(ProxyStreamEvent::Open) => {
            // The join handle is dropped here on purpose: **the connection
            // ends the answers, never the work.** A turn already upstream is
            // paid for, so a caller that disappears loses its delivery and
            // nothing else — the same rule every other transport in this app
            // keeps.
            drop(streaming);
            let stream = futures_util::stream::unfold(rx, |mut rx| async move {
                loop {
                    match rx.recv().await {
                        Some(ProxyStreamEvent::Chunk(bytes)) => {
                            return Some((
                                Ok::<_, Infallible>(Frame::data(Bytes::from(bytes))),
                                rx,
                            ));
                        }
                        // `Open` is sent exactly once, before any chunk, so
                        // this is unreachable in practice; ignoring it beats
                        // ending the body on a frame that means nothing here.
                        Some(ProxyStreamEvent::Open) => continue,
                        None => return None,
                    }
                }
            });
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .header(CACHE_CONTROL, "no-store")
                .body(BoxBody::new(StreamBody::new(stream)))
                .expect("a response with a valid status and headers")
        }
        // The channel closed with no `Open`: the turn never reached the
        // upstream, so its error is the whole answer.
        _ => match streaming.await {
            Ok(Err(e)) => app_error_response(&e),
            Ok(Ok(())) => error_response(
                StatusCode::BAD_GATEWAY,
                "server_error",
                "The upstream closed before answering.",
                None,
            ),
            Err(_) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "The request could not be completed.",
                None,
            ),
        },
    }
}

/// Map a typed failure onto the OpenAI error shape.
///
/// **Routing is on the variant, never on a message** — the same rule the GUI's
/// error copy follows. What a caller can do about a failure is a property of
/// its kind: a funding problem is the reader's to fix in Eidola, an unexposed
/// model is the tool's request to change, and an upstream's own status is
/// passed through so a client's retry logic sees what really happened.
pub(crate) fn app_error_response(error: &AppError) -> Response<ProxyBody> {
    let (status, kind, code) = match error {
        AppError::ModelUnavailable { .. } => (
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            Some("model_not_found"),
        ),
        AppError::Config { .. } => (StatusCode::BAD_REQUEST, "invalid_request_error", None),
        AppError::NotConfigured { .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            "invalid_request_error",
            None,
        ),
        AppError::NoAccount
        | AppError::InsufficientBalance { .. }
        | AppError::Credential { .. }
        | AppError::ProvisioningTimeout { .. } => (
            StatusCode::PAYMENT_REQUIRED,
            "insufficient_quota",
            Some("insufficient_quota"),
        ),
        // An upstream status is passed through where it is a status at all —
        // a client's backoff should see the 429 the server sent, not a 502
        // this app invented over it.
        AppError::Server { status, .. } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            "server_error",
            None,
        ),
        AppError::Network { .. } | AppError::Attestation { .. } => {
            (StatusCode::BAD_GATEWAY, "server_error", None)
        }
        AppError::LocalModel { .. } => (StatusCode::SERVICE_UNAVAILABLE, "server_error", None),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", None),
    };
    error_response(status, kind, &error.to_string(), code)
}

fn error_response(
    status: StatusCode,
    kind: &str,
    message: &str,
    code: Option<&str>,
) -> Response<ProxyBody> {
    json_response(
        status,
        json!({
            "error": {
                "message": message,
                "type": kind,
                "code": code,
            }
        }),
    )
}

fn json_response(status: StatusCode, body: Value) -> Response<ProxyBody> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(BoxBody::new(
            Full::new(Bytes::from(body.to_string())).map_err(|never| match never {}),
        ))
        .expect("a response with a valid status and headers")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_auth(value: Option<&str>) -> Request<()> {
        let mut builder = Request::builder().uri("/v1/models");
        if let Some(value) = value {
            builder = builder.header(AUTHORIZATION, value);
        }
        builder.body(()).expect("request")
    }

    #[test]
    fn a_bearer_token_is_read_and_nothing_else_is() {
        assert_eq!(
            bearer_token(&request_with_auth(Some("Bearer eid-abc"))),
            Some("eid-abc".to_string())
        );
        assert_eq!(
            bearer_token(&request_with_auth(Some("bearer eid-abc"))),
            Some("eid-abc".to_string()),
            "the scheme is case-insensitive, as HTTP says"
        );
        assert_eq!(bearer_token(&request_with_auth(Some("Basic abc"))), None);
        assert_eq!(bearer_token(&request_with_auth(Some("Bearer   "))), None);
        assert_eq!(bearer_token(&request_with_auth(None)), None);
    }

    #[test]
    fn a_model_id_survives_the_path() {
        assert_eq!(percent_decode("gemma%40local"), "gemma@local");
        assert_eq!(percent_decode("org%2Fmodel"), "org/model");
        assert_eq!(percent_decode("plain"), "plain");
        assert_eq!(
            percent_decode("%zz"),
            "%zz",
            "an undecodable escape is left alone and simply matches nothing"
        );
    }

    #[test]
    fn a_models_entry_names_the_backend_it_is_reached_through() {
        let entry = model_entry(&crate::ModelInfo {
            id: "gemma@local".into(),
            context_length: 0,
            max_output_tokens: None,
            output_budget_class: None,
            capabilities: Default::default(),
            prompt_credits_per_token: 0.0,
            completion_credits_per_token: 0.0,
            request_credits: None,
        });
        assert_eq!(entry["id"], "gemma@local");
        assert_eq!(entry["object"], "model");
        assert_eq!(entry["owned_by"], "local");
    }

    #[test]
    fn failures_route_on_the_variant() {
        let status = |e: AppError| app_error_response(&e).status();
        assert_eq!(
            status(AppError::ModelUnavailable { model: "m".into() }),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            status(AppError::Config {
                message: "bad".into()
            }),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            status(AppError::InsufficientBalance {
                available: 0,
                required: 1,
            }),
            StatusCode::PAYMENT_REQUIRED
        );
        assert_eq!(
            status(AppError::Server {
                status: 429,
                message: "slow down".into()
            }),
            StatusCode::TOO_MANY_REQUESTS,
            "a client's backoff sees what the server really said"
        );
        assert_eq!(
            status(AppError::Network {
                message: "gone".into()
            }),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn an_error_body_is_the_shape_an_openai_client_reads() {
        let response = error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "nope",
            Some("model_not_found"),
        );
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }
}
