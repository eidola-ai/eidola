//! HTTP/1.1 over a stream the host dials ([`super::HostNetwork`]).
//!
//! Each request dials a fresh stream, runs TLS over it — and for the Eidola
//! backend the attestation check, before a byte of the request is written
//! (`tinfoil_verifier::pipe`) — then speaks HTTP/1.1 through hyper's client
//! connection, whose task runs on [`crate::rt`] until the response is read or
//! dropped. Nothing is pooled, so every request is attested on its own
//! connection.
//!
//! What it carries matches the native client: methods, headers in the order
//! given (with reqwest's defaults — `Accept: */*`, the client's `User-Agent` —
//! added where absent), a body known up front or produced when first asked
//! for, a streamed response, a whole-request timeout, and reqwest's redirect
//! rules where the client follows redirects.

use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http::header::{self, HeaderMap, HeaderValue};
use http::{Method, StatusCode};
use hyper::body::{Frame, SizeHint};
use tinfoil_verifier::pipe::{ATTESTATION_DEADLINE, AttestingPipe, Dialer, PipeStream, TlsPipe};

use super::{Body, BodyKind, BodyStream, Error, HostNetwork, Redirects};
use crate::rt;

/// The most redirects a following client takes, as reqwest's default policy.
const MAX_REDIRECTS: usize = 10;

pub(super) struct PipeClient {
    pub(super) dialer: Arc<dyn Dialer>,
    pub(super) tls: PipeTls,
    pub(super) user_agent: Option<&'static str>,
    pub(super) redirects: Redirects,
    /// Bounds dialing and the TLS handshake.
    pub(super) connect_timeout: Option<Duration>,
}

/// How a client's connections are secured.
pub(super) enum PipeTls {
    /// The Eidola backend: TLS, then the attestation check, on every
    /// connection. Only `https` URLs.
    Attesting(AttestingPipe),
    /// TLS to the trust store for `https` URLs; `http` URLs in the clear.
    Plain(TlsPipe),
}

impl PipeClient {
    pub(super) fn plain(
        host: &HostNetwork,
        user_agent: Option<&'static str>,
        redirects: Redirects,
        connect_timeout: Option<Duration>,
    ) -> Self {
        Self {
            dialer: host.dialer.clone(),
            tls: PipeTls::Plain(TlsPipe::new(super::roots_with(host))),
            user_agent,
            redirects,
            connect_timeout,
        }
    }

    /// Send `request`, following redirects if this client does, bounded by
    /// its timeout from here until its body has been read.
    pub(super) async fn execute(&self, request: PipeRequest) -> Result<PipeResponse, Error> {
        let deadline = request.timeout.map(|timeout| rt::Instant::now() + timeout);
        let sent = self.follow(request);
        let mut response = match deadline {
            Some(deadline) => rt::timeout_at(deadline, sent)
                .await
                .map_err(|_| Error::request("operation timed out"))??,
            None => sent.await?,
        };
        response.deadline = deadline;
        Ok(response)
    }

    /// reqwest's redirect rules: up to [`MAX_REDIRECTS`]; `301`/`302`/`303`
    /// drop the body and its headers and turn anything but `GET`/`HEAD` into
    /// `GET`; `307`/`308` resend the body, and are handed back as the answer
    /// when it cannot be resent (a deferred body is produced once); a hop to
    /// another origin drops the credentials. No `Referer` is added: it would
    /// carry the previous URL, query and all, to the next host.
    async fn follow(&self, mut request: PipeRequest) -> Result<PipeResponse, Error> {
        let mut hops = 0;
        loop {
            let resend = match self.redirects {
                Redirects::Follow => request.body.try_clone(),
                Redirects::Refuse => None,
            };
            let previous = (
                request.method.clone(),
                request.url.clone(),
                request.headers.clone(),
                request.timeout,
            );
            let response = self.send_once(request).await?;
            if self.redirects == Redirects::Refuse || !response.status.is_redirection() {
                return Ok(response);
            }
            let (method, url, mut headers, timeout) = previous;
            let next = response
                .headers
                .get(header::LOCATION)
                .and_then(|location| location.to_str().ok())
                .and_then(|location| url.join(location).ok())
                .filter(|next| matches!(next.scheme(), "http" | "https"));
            let Some(next) = next else {
                return Ok(response);
            };
            let (method, body) = match response.status {
                StatusCode::MOVED_PERMANENTLY | StatusCode::FOUND | StatusCode::SEE_OTHER => {
                    for name in [
                        header::TRANSFER_ENCODING,
                        header::CONTENT_ENCODING,
                        header::CONTENT_TYPE,
                        header::CONTENT_LENGTH,
                    ] {
                        headers.remove(name);
                    }
                    let method = match method {
                        Method::GET | Method::HEAD => method,
                        _ => Method::GET,
                    };
                    (method, Body::empty())
                }
                StatusCode::TEMPORARY_REDIRECT | StatusCode::PERMANENT_REDIRECT => match resend {
                    Some(body) => (method, body),
                    None => return Ok(response),
                },
                _ => return Ok(response),
            };
            hops += 1;
            if hops > MAX_REDIRECTS {
                return Err(Error::request("too many redirects"));
            }
            if (next.scheme(), next.host_str(), next.port_or_known_default())
                != (url.scheme(), url.host_str(), url.port_or_known_default())
            {
                for name in [
                    header::AUTHORIZATION,
                    header::COOKIE,
                    header::PROXY_AUTHORIZATION,
                    header::WWW_AUTHENTICATE,
                ] {
                    headers.remove(name);
                }
                headers.remove("cookie2");
            }
            drop(response);
            request = PipeRequest {
                method,
                url: next,
                headers,
                body,
                timeout,
            };
        }
    }

    /// One request on its own connection.
    async fn send_once(&self, request: PipeRequest) -> Result<PipeResponse, Error> {
        let PipeRequest {
            method,
            url,
            headers,
            body,
            ..
        } = request;
        let host = url
            .host_str()
            .ok_or_else(|| Error::builder("URL has no host"))?
            .to_string();
        let port = url
            .port_or_known_default()
            .ok_or_else(|| Error::builder("URL has no port"))?;
        let stream = self.connect(url.scheme() == "https", &host, port).await?;

        let (mut sender, connection) = hyper::client::conn::http1::handshake::<_, OutBody>(
            hyper_util::rt::TokioIo::new(stream),
        )
        .await
        .map_err(|e| Error::request(format!("HTTP handshake: {e}")))?;
        // The connection runs until its response is read; dropping the
        // response (or its stream) drops `close`, which ends it early.
        let (close, closed) = tokio::sync::oneshot::channel::<()>();
        rt::spawn(async move {
            let connection = std::pin::pin!(connection);
            let _ = futures_util::future::select(connection, closed).await;
        });

        // `Host` first, as a client writes it; then the request's own
        // headers in order; then the client's defaults where absent.
        let authority = match url.port() {
            Some(port) => format!("{host}:{port}"),
            None => host.clone(),
        };
        let mut head = HeaderMap::with_capacity(headers.len() + 3);
        head.insert(
            header::HOST,
            HeaderValue::try_from(authority).map_err(|e| Error::builder(format!("host: {e}")))?,
        );
        // Iteration names a header once, before all of its values.
        let mut name = None;
        for (next, value) in headers {
            if next.is_some() {
                name = next;
            }
            if let Some(name) = &name {
                head.append(name.clone(), value);
            }
        }
        if !head.contains_key(header::ACCEPT) {
            head.insert(header::ACCEPT, HeaderValue::from_static("*/*"));
        }
        if let Some(agent) = self.user_agent
            && !head.contains_key(header::USER_AGENT)
        {
            head.insert(header::USER_AGENT, HeaderValue::from_static(agent));
        }

        let path = match url.query() {
            Some(query) => format!("{}?{query}", url.path()),
            None => url.path().to_string(),
        };
        let mut outbound = http::Request::new(OutBody(Some(body.kind)));
        *outbound.method_mut() = method;
        *outbound.uri_mut() = path
            .parse()
            .map_err(|e| Error::builder(format!("request target: {e}")))?;
        *outbound.headers_mut() = head;

        let response = sender
            .send_request(outbound)
            .await
            .map_err(|e| Error::request(format!("{e}{}", source_text(&e))))?;
        let (parts, body) = response.into_parts();
        Ok(PipeResponse {
            status: parts.status,
            headers: parts.headers,
            body,
            _close: close,
            deadline: None,
        })
    }

    /// Dial `host:port` through the host and secure it as this client does.
    async fn connect(&self, https: bool, host: &str, port: u16) -> Result<PipeStream, Error> {
        match (&self.tls, https) {
            (PipeTls::Attesting(pipe), true) => {
                let handshaken = self
                    .bounded(pipe.handshake(self.dialer.as_ref(), host, port))
                    .await?;
                // The check never reads a clock; this is the one it is given.
                let now = rt::SystemTime::now()
                    .duration_since(rt::UNIX_EPOCH)
                    .map_err(|e| Error::request(format!("system clock: {e}")))?
                    .as_secs();
                match rt::timeout(ATTESTATION_DEADLINE, handshaken.attest(now)).await {
                    Ok(Ok(stream)) => Ok(Box::new(stream)),
                    Ok(Err(e)) => Err(Error::request(format!("attestation failed: {e}"))),
                    Err(_) => Err(Error::request(format!(
                        "attestation failed: {}",
                        tinfoil_verifier::Error::AttestationTimeout {
                            seconds: ATTESTATION_DEADLINE.as_secs(),
                        }
                    ))),
                }
            }
            (PipeTls::Attesting(_), false) => Err(Error::request(
                "attestation failed: an attested endpoint is reached over TLS, and this URL is not https",
            )),
            (PipeTls::Plain(tls), true) => self
                .bounded(tls.connect(self.dialer.as_ref(), host, port))
                .await
                .map(|stream| Box::new(stream) as PipeStream),
            (PipeTls::Plain(_), false) => {
                self.bounded(async {
                    self.dialer
                        .dial(host, port)
                        .await
                        .map_err(|e| tinfoil_verifier::Error::Connect(format!("dial: {e}")))
                })
                .await
            }
        }
    }

    /// `step` within the connect timeout, its error as this transport's.
    async fn bounded<T>(
        &self,
        step: impl Future<Output = Result<T, tinfoil_verifier::Error>>,
    ) -> Result<T, Error> {
        let done = match self.connect_timeout {
            Some(timeout) => rt::timeout(timeout, step)
                .await
                .map_err(|_| Error::request("connection timed out"))?,
            None => step.await,
        };
        done.map_err(|e| Error::request(e.to_string()))
    }
}

/// The causes hyper keeps behind its own message.
fn source_text(e: &(dyn std::error::Error + 'static)) -> String {
    let mut text = String::new();
    let mut source = e.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// A request as the facade builds it.
pub(super) struct PipeRequest {
    pub(super) method: Method,
    pub(super) url: url::Url,
    pub(super) headers: HeaderMap,
    pub(super) body: Body,
    pub(super) timeout: Option<Duration>,
}

impl PipeRequest {
    /// A request for `url`, parsed as reqwest parses it, its userinfo
    /// already split off (`super::split_userinfo`).
    pub(super) fn new(method: Method, url: url::Url) -> Result<Self, Error> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(Error::builder(format!(
                "URL scheme is not allowed: {}",
                url.scheme()
            )));
        }
        Ok(Self {
            method,
            url,
            headers: HeaderMap::new(),
            body: Body::empty(),
            timeout: None,
        })
    }
}

/// A response whose body is still on the wire.
pub(super) struct PipeResponse {
    pub(super) status: StatusCode,
    pub(super) headers: HeaderMap,
    body: hyper::body::Incoming,
    /// Dropped with the response, which ends its connection.
    _close: tokio::sync::oneshot::Sender<()>,
    /// The request's timeout, which the body is read within.
    deadline: Option<rt::Instant>,
}

impl PipeResponse {
    pub(super) fn content_length(&self) -> Option<u64> {
        hyper::body::Body::size_hint(&self.body).exact()
    }

    pub(super) fn into_stream(self) -> BodyStream {
        Box::pin(futures_util::stream::unfold(
            Some(self),
            |state| async move {
                let mut response = state?;
                loop {
                    let next = std::future::poll_fn(|cx| {
                        hyper::body::Body::poll_frame(std::pin::Pin::new(&mut response.body), cx)
                    });
                    let frame = match response.deadline {
                        Some(deadline) => match rt::timeout_at(deadline, next).await {
                            Ok(frame) => frame,
                            Err(_) => {
                                return Some((Err(Error::request("operation timed out")), None));
                            }
                        },
                        None => next.await,
                    };
                    match frame {
                        None => return None,
                        Some(Err(e)) => {
                            let error = Error::request(format!(
                                "reading the response body: {e}{}",
                                source_text(&e)
                            ));
                            return Some((Err(error), None));
                        }
                        Some(Ok(frame)) => {
                            // Trailers carry nothing this crate reads.
                            if let Ok(data) = frame.into_data()
                                && !data.is_empty()
                            {
                                return Some((Ok(data), Some(response)));
                            }
                        }
                    }
                }
            },
        ))
    }
}

/// A request body as hyper reads it: the bytes, or the deferred chunk polled
/// when hyper first asks — after the connection is open (and attested).
struct OutBody(Option<BodyKind>);

impl hyper::body::Body for OutBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, std::io::Error>>> {
        match self.0.take() {
            None => Poll::Ready(None),
            Some(BodyKind::Bytes(bytes)) if bytes.is_empty() => Poll::Ready(None),
            Some(BodyKind::Bytes(bytes)) => Poll::Ready(Some(Ok(Frame::data(bytes)))),
            Some(BodyKind::Deferred(mut chunk)) => match chunk.as_mut().poll(cx) {
                Poll::Ready(Ok(bytes)) => Poll::Ready(Some(Ok(Frame::data(bytes)))),
                Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
                Poll::Pending => {
                    self.0 = Some(BodyKind::Deferred(chunk));
                    Poll::Pending
                }
            },
        }
    }

    fn is_end_stream(&self) -> bool {
        match &self.0 {
            None => true,
            Some(BodyKind::Bytes(bytes)) => bytes.is_empty(),
            Some(BodyKind::Deferred(_)) => false,
        }
    }

    fn size_hint(&self) -> SizeHint {
        match &self.0 {
            None => SizeHint::with_exact(0),
            Some(BodyKind::Bytes(bytes)) => SizeHint::with_exact(bytes.len() as u64),
            Some(BodyKind::Deferred(_)) => SizeHint::default(),
        }
    }
}
