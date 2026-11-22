//! Every HTTP request this crate makes goes through here.
//!
//! [`Client`] is one facade over two transports:
//!
//! - **native** (`reqwest`): the attested client the Eidola backend is reached
//!   through (`tinfoil_verifier::attesting_client`), the plain client for
//!   engines and external backends, and the updater's, all over rustls with
//!   the OS trust store. What every desktop build uses.
//! - **the host's socket** ([`pipe`]): when the embedding app supplies a
//!   [`Dialer`] ([`HostNetwork`], given to `AppCore::with_host_network`), every
//!   request is a stream that dialer opens, with TLS, and for the Eidola
//!   backend the attestation check, run inside this process
//!   (`tinfoil_verifier::pipe`). This is how the mobile app's wasm build
//!   reaches the network: its shell moves ciphertext and decides nothing.
//!
//! wasm32 has no native transport and links no `reqwest`, so nothing there can
//! fall back to the browser's `fetch`: without a dialer every builder refuses
//! with `AppError::NoTransport`.
//!
//! Both transports call [`crate::durability::barrier`] before a request leaves
//! (in [`RequestBuilder::send`] and [`Client::execute`], the only doors out),
//! and again after a deferred body produces its bytes and before they are
//! written ([`deferred_body`]).

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
#[cfg(not(target_family = "wasm"))]
use futures_util::StreamExt as _;
use http::header::{self, HeaderMap, HeaderName, HeaderValue};
use http::{Method, StatusCode};

use crate::error::AppError;

mod pipe;
#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

#[cfg(not(target_family = "wasm"))]
pub use tinfoil_verifier::pipe::TcpDialer;
pub use tinfoil_verifier::pipe::{BoxFuture, ByteStream, Dialer, PipeStream};

/// The socket an embedding app provides: every request this core makes is a
/// stream [`HostNetwork::dialer`] opens.
#[derive(Clone)]
pub struct HostNetwork {
    /// Opens raw byte streams; TLS and every trust decision stay in this
    /// process.
    pub dialer: Arc<dyn Dialer>,
    /// TLS roots trusted beside the built-in ones (the OS store natively, the
    /// Mozilla roots on wasm32) — what a developer's keychain does natively,
    /// for a development server's CA. Attestation is unaffected: it binds the
    /// attested key whatever root the certificate chains to.
    pub extra_tls_roots: Vec<rustls::pki_types::CertificateDer<'static>>,
}

/// How one core reaches the network: through its host's dialer when it has
/// one, natively otherwise.
#[derive(Clone, Default)]
pub(crate) struct Transport {
    host: Option<HostNetwork>,
}

impl Transport {
    pub(crate) fn new(host: Option<HostNetwork>) -> Self {
        Self { host }
    }
}

/// Whether a client follows a redirect, or hands it back as the answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Redirects {
    Follow,
    Refuse,
}

/// Bounds the TCP connect and TLS handshake of the attested client.
const ATTESTED_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Bounds the handshake of the updater's client. A whole-request timeout would
/// be wrong there: the same client fetches a small signed document and a
/// container of tens of megabytes, and the second is allowed to take as long
/// as it keeps making progress — which is what the per-read stall bound
/// measures.
const UPDATE_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

const USER_AGENT: &str = concat!("eidola-app-core/", env!("CARGO_PKG_VERSION"));

/// The client for the Eidola backend: every new connection attested against
/// `allowed_measurements` (`tinfoil_verifier`), no redirect followed.
pub(crate) async fn attesting_client(
    transport: &Transport,
    allowed_measurements: &[tinfoil_verifier::AllowedMeasurement],
    base_url: &str,
    hardware_root_der: Option<&[u8]>,
    hardware_intermediate_der: Option<&[u8]>,
    attestation_observer: Option<tinfoil_verifier::AttestationObserver>,
) -> Result<Client, AppError> {
    let config = |tls_roots| tinfoil_verifier::AttestingClientConfig {
        allowed_measurements,
        inference_base_url: base_url,
        trusted_ark_der: hardware_root_der,
        trusted_ask_der: hardware_intermediate_der,
        snp_min_tcb: None,
        snp_observer: None,
        attestation_observer,
        tls_roots,
    };
    let built = match &transport.host {
        Some(host) => {
            tinfoil_verifier::pipe::AttestingPipe::new(config(roots_with(host))).map(|tls| {
                Client::pipe(pipe::PipeClient {
                    dialer: host.dialer.clone(),
                    tls: pipe::PipeTls::Attesting(tls),
                    user_agent: None,
                    redirects: Redirects::Refuse,
                    connect_timeout: Some(ATTESTED_CONNECT_TIMEOUT),
                })
            })
        }
        #[cfg(not(target_family = "wasm"))]
        None => tinfoil_verifier::attesting_client(config(native::root_store()))
            .await
            .map(Client::native),
        #[cfg(target_family = "wasm")]
        None => return Err(AppError::NoTransport),
    };
    built.map_err(|e| AppError::Attestation {
        message: format!("attestation client build failed: {e}"),
    })
}

/// A plain (non-attesting) HTTPS-capable client, for local engines and
/// external backends (`local_models::plain_http_client` and its siblings).
pub(crate) fn plain_client(
    transport: &Transport,
    user_agent: Option<&'static str>,
    redirects: Redirects,
) -> Result<Client, AppError> {
    match &transport.host {
        Some(host) => Ok(Client::pipe(pipe::PipeClient::plain(
            host, user_agent, redirects, None,
        ))),
        #[cfg(not(target_family = "wasm"))]
        None => native::plain_client(user_agent, redirects)
            .map(Client::native)
            .map_err(|e| AppError::LocalModel {
                message: format!("constructing HTTP client: {e}"),
            }),
        #[cfg(target_family = "wasm")]
        None => Err(AppError::NoTransport),
    }
}

/// The updater's client (`updater::build_http_client`): follows redirects
/// (release assets are served from a CDN), with a bounded handshake.
pub(crate) fn update_client(transport: &Transport) -> Result<Client, AppError> {
    match &transport.host {
        Some(host) => Ok(Client::pipe(pipe::PipeClient::plain(
            host,
            Some(USER_AGENT),
            Redirects::Follow,
            Some(UPDATE_CONNECT_TIMEOUT),
        ))),
        #[cfg(not(target_family = "wasm"))]
        None => native::update_client()
            .map(Client::native)
            .map_err(|e| AppError::Update {
                message: format!("constructing HTTPS client: {e}"),
            }),
        #[cfg(target_family = "wasm")]
        None => Err(AppError::NoTransport),
    }
}

/// A username and an optional password.
type Credentials = (String, Option<String>);

/// `url` without its userinfo, and the credentials it held: reqwest's rule
/// (`extract_authority`), so both transports send the same head. Userinfo
/// that is not UTF-8 once percent-decoded is left where it is, as reqwest
/// leaves it; the host-socket transport never writes userinfo out.
fn split_userinfo(url: &str) -> (Result<url::Url, url::ParseError>, Option<Credentials>) {
    let mut parsed = match url::Url::parse(url) {
        Ok(parsed) => parsed,
        Err(e) => return (Err(e), None),
    };
    let decode = |part: &str| {
        percent_encoding::percent_decode_str(part)
            .decode_utf8()
            .ok()
            .map(String::from)
    };
    if !parsed.has_authority() {
        return (Ok(parsed), None);
    }
    let Some(username) = decode(parsed.username()) else {
        return (Ok(parsed), None);
    };
    let password = parsed.password().and_then(decode);
    if username.is_empty() && password.is_none() {
        return (Ok(parsed), None);
    }
    // An authority always takes an empty userinfo.
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    (Ok(parsed), Some((username, password)))
}

/// The trust store a host-dialed TLS connection is checked against: the
/// built-in roots, and the host's extras.
fn roots_with(host: &HostNetwork) -> rustls::RootCertStore {
    #[cfg(not(target_family = "wasm"))]
    let mut store = native::root_store();
    #[cfg(target_family = "wasm")]
    let mut store = rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    for cert in &host.extra_tls_roots {
        let _ = store.add(cert.clone());
    }
    store
}

#[cfg(not(target_family = "wasm"))]
mod native {
    use super::{Redirects, UPDATE_CONNECT_TIMEOUT, USER_AGENT};

    pub(super) fn plain_client(
        user_agent: Option<&'static str>,
        redirects: Redirects,
    ) -> reqwest::Result<reqwest::Client> {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store())
            .with_no_client_auth();
        let mut builder = reqwest::Client::builder().tls_backend_preconfigured(tls_config);
        if let Some(agent) = user_agent {
            builder = builder.user_agent(agent);
        }
        if redirects == Redirects::Refuse {
            builder = builder.redirect(reqwest::redirect::Policy::none());
        }
        builder.build()
    }

    pub(super) fn update_client() -> reqwest::Result<reqwest::Client> {
        // Idempotent — AppCore::new normally installs this; for standalone
        // updater callers (tests, future binaries) this is the safety net.
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store())
            .with_no_client_auth();
        reqwest::Client::builder()
            .tls_backend_preconfigured(tls_config)
            .user_agent(USER_AGENT)
            .connect_timeout(UPDATE_CONNECT_TIMEOUT)
            .build()
    }

    pub(super) fn root_store() -> rustls::RootCertStore {
        let mut store = rustls::RootCertStore::empty();
        let native = rustls_native_certs::load_native_certs();
        for cert in native.certs {
            let _ = store.add(cert);
        }
        store
    }
}

// ---------------------------------------------------------------------------
// The facade
// ---------------------------------------------------------------------------

/// An HTTP client, on whichever transport built it.
#[derive(Clone)]
pub struct Client {
    kind: ClientKind,
}

#[derive(Clone)]
enum ClientKind {
    #[cfg(not(target_family = "wasm"))]
    Native(reqwest::Client),
    Pipe(Arc<pipe::PipeClient>),
}

#[cfg(not(target_family = "wasm"))]
impl From<reqwest::Client> for Client {
    fn from(client: reqwest::Client) -> Self {
        Self::native(client)
    }
}

impl Client {
    #[cfg(not(target_family = "wasm"))]
    fn native(client: reqwest::Client) -> Self {
        Self {
            kind: ClientKind::Native(client),
        }
    }

    fn pipe(client: pipe::PipeClient) -> Self {
        Self {
            kind: ClientKind::Pipe(Arc::new(client)),
        }
    }

    /// **Test-only seam.** A plain client over `dialer` that follows
    /// redirects, as reqwest's default client does: what a test injects to
    /// run the crate's requests over the host-socket transport against a
    /// plain-HTTP mock.
    #[doc(hidden)]
    #[cfg(feature = "test-support")]
    pub fn test_plain_over(dialer: Arc<dyn Dialer>) -> Self {
        let host = HostNetwork {
            dialer,
            extra_tls_roots: Vec::new(),
        };
        Self::pipe(pipe::PipeClient::plain(
            &host,
            None,
            Redirects::Follow,
            None,
        ))
    }

    pub(crate) fn get(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::GET, url.as_ref())
    }

    pub(crate) fn post(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::POST, url.as_ref())
    }

    /// A request for `url`. Credentials in its userinfo are split off here,
    /// for both transports, as reqwest splits them: percent-decoded into a
    /// sensitive `Authorization: Basic` (appended first, so an explicit
    /// `Authorization` is appended after it, as reqwest appends), and removed
    /// from the URL, so neither transport writes them into the request target
    /// or `Host`. A redirect to another origin drops them with the header.
    fn request(&self, method: Method, url: &str) -> RequestBuilder {
        let (parsed, credentials) = split_userinfo(url);
        let kind = match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            ClientKind::Native(client) => BuilderKind::Native(match parsed {
                Ok(parsed) => client.request(method, parsed),
                // reqwest refuses it in its own words.
                Err(_) => client.request(method, url),
            }),
            ClientKind::Pipe(client) => BuilderKind::Pipe {
                client: client.clone(),
                request: parsed
                    .map_err(|e| Error::builder(format!("URL: {e}")))
                    .and_then(|parsed| pipe::PipeRequest::new(method, parsed)),
            },
        };
        let builder = RequestBuilder { kind };
        match credentials {
            Some((username, password)) => builder.basic_auth(username, password),
            None => builder,
        }
    }

    /// Send a built request. A request built by another client is refused.
    pub(crate) async fn execute(&self, request: Request) -> Result<Response, Error> {
        crate::durability::barrier().await;
        match (&self.kind, request.kind) {
            #[cfg(not(target_family = "wasm"))]
            (ClientKind::Native(client), RequestKind::Native(request)) => client
                .execute(request)
                .await
                .map(Response::from)
                .map_err(Error::from),
            (ClientKind::Pipe(client), RequestKind::Pipe(request)) => {
                client.execute(request).await.map(Response::pipe)
            }
            #[cfg(not(target_family = "wasm"))]
            _ => Err(Error::builder("the request was built by another client")),
        }
    }
}

/// A request under construction. The methods are reqwest's, and so are their
/// semantics on both transports: `header` appends, `json` sets a
/// `Content-Type` only where none is set, the auth helpers mark their header
/// sensitive, and a refused name or value fails the request at `build`/`send`.
pub(crate) struct RequestBuilder {
    kind: BuilderKind,
}

enum BuilderKind {
    #[cfg(not(target_family = "wasm"))]
    Native(reqwest::RequestBuilder),
    Pipe {
        client: Arc<pipe::PipeClient>,
        request: Result<pipe::PipeRequest, Error>,
    },
}

impl RequestBuilder {
    fn map_pipe(
        self,
        f: impl FnOnce(pipe::PipeRequest) -> Result<pipe::PipeRequest, Error>,
    ) -> Self {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(_) => self,
            BuilderKind::Pipe { client, request } => RequestBuilder {
                kind: BuilderKind::Pipe {
                    client,
                    request: request.and_then(f),
                },
            },
        }
    }

    pub(crate) fn header<K, V>(self, name: K, value: V) -> Self
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => RequestBuilder {
                kind: BuilderKind::Native(builder.header(name, value)),
            },
            BuilderKind::Pipe { .. } => self.map_pipe(|mut request| {
                let name = HeaderName::try_from(name)
                    .map_err(|e| Error::builder(format!("header name: {}", e.into())))?;
                let value = HeaderValue::try_from(value)
                    .map_err(|e| Error::builder(format!("header value: {}", e.into())))?;
                request.headers.append(name, value);
                Ok(request)
            }),
        }
    }

    pub(crate) fn basic_auth(
        self,
        username: impl std::fmt::Display,
        password: Option<impl std::fmt::Display>,
    ) -> Self {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => RequestBuilder {
                kind: BuilderKind::Native(builder.basic_auth(username, password)),
            },
            BuilderKind::Pipe { .. } => {
                use base64::Engine as _;
                let credentials = match password {
                    Some(password) => format!("{username}:{password}"),
                    None => format!("{username}:"),
                };
                let encoded = base64::engine::general_purpose::STANDARD.encode(credentials);
                self.sensitive(header::AUTHORIZATION, format!("Basic {encoded}"))
            }
        }
    }

    pub(crate) fn bearer_auth(self, token: impl std::fmt::Display) -> Self {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => RequestBuilder {
                kind: BuilderKind::Native(builder.bearer_auth(token)),
            },
            BuilderKind::Pipe { .. } => {
                self.sensitive(header::AUTHORIZATION, format!("Bearer {token}"))
            }
        }
    }

    fn sensitive(self, name: HeaderName, value: String) -> Self {
        self.map_pipe(|mut request| {
            let mut value = HeaderValue::try_from(value)
                .map_err(|e| Error::builder(format!("header value: {e}")))?;
            value.set_sensitive(true);
            request.headers.append(name, value);
            Ok(request)
        })
    }

    pub(crate) fn json<T: serde::Serialize + ?Sized>(self, json: &T) -> Self {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => RequestBuilder {
                kind: BuilderKind::Native(builder.json(json)),
            },
            BuilderKind::Pipe { .. } => {
                let serialized = serde_json::to_vec(json);
                self.map_pipe(|mut request| {
                    let bytes = serialized.map_err(|e| Error::builder(format!("json: {e}")))?;
                    if !request.headers.contains_key(header::CONTENT_TYPE) {
                        request.headers.insert(
                            header::CONTENT_TYPE,
                            HeaderValue::from_static("application/json"),
                        );
                    }
                    request.body = Body::from(bytes);
                    Ok(request)
                })
            }
        }
    }

    pub(crate) fn body(self, body: impl Into<Body>) -> Self {
        let body = body.into();
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => RequestBuilder {
                kind: BuilderKind::Native(builder.body(body.into_native())),
            },
            BuilderKind::Pipe { .. } => self.map_pipe(|mut request| {
                request.body = body;
                Ok(request)
            }),
        }
    }

    /// Bound the whole request, from connecting until its body has been read.
    pub(crate) fn timeout(self, timeout: Duration) -> Self {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => RequestBuilder {
                kind: BuilderKind::Native(builder.timeout(timeout)),
            },
            BuilderKind::Pipe { .. } => self.map_pipe(|mut request| {
                request.timeout = Some(timeout);
                Ok(request)
            }),
        }
    }

    pub(crate) fn build(self) -> Result<Request, Error> {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => builder
                .build()
                .map(|request| Request {
                    kind: RequestKind::Native(request),
                })
                .map_err(Error::from),
            BuilderKind::Pipe { request, .. } => request.map(|request| Request {
                kind: RequestKind::Pipe(request),
            }),
        }
    }

    pub(crate) async fn send(self) -> Result<Response, Error> {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            BuilderKind::Native(builder) => {
                crate::durability::barrier().await;
                builder
                    .send()
                    .await
                    .map(Response::from)
                    .map_err(Error::from)
            }
            BuilderKind::Pipe { client, request } => {
                let request = request?;
                crate::durability::barrier().await;
                client.execute(request).await.map(Response::pipe)
            }
        }
    }
}

/// A built request, for [`Client::execute`].
pub(crate) struct Request {
    kind: RequestKind,
}

enum RequestKind {
    #[cfg(not(target_family = "wasm"))]
    Native(reqwest::Request),
    Pipe(pipe::PipeRequest),
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            RequestKind::Native(request) => request.fmt(f),
            RequestKind::Pipe(request) => f
                .debug_struct("Request")
                .field("method", &request.method)
                .field("headers", &request.headers)
                .finish_non_exhaustive(),
        }
    }
}

impl Request {
    #[cfg(test)]
    pub(crate) fn headers(&self) -> &HeaderMap {
        match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            RequestKind::Native(request) => request.headers(),
            RequestKind::Pipe(request) => &request.headers,
        }
    }

    /// The proxy replaces a built request's headers with its allowlist.
    #[cfg(feature = "desktop")]
    pub(crate) fn headers_mut(&mut self) -> &mut HeaderMap {
        match &mut self.kind {
            #[cfg(not(target_family = "wasm"))]
            RequestKind::Native(request) => request.headers_mut(),
            RequestKind::Pipe(request) => &mut request.headers,
        }
    }
}

/// A request body: bytes known up front, or one chunk produced when the
/// transport first asks for it ([`deferred_body`]).
pub(crate) struct Body {
    kind: BodyKind,
}

pub(crate) type DeferredChunk = BoxFuture<'static, std::io::Result<Bytes>>;

enum BodyKind {
    Bytes(Bytes),
    Deferred(DeferredChunk),
}

impl Body {
    pub(super) fn empty() -> Self {
        Self::from(Bytes::new())
    }

    /// The bytes again, for a redirect that resends them; a deferred body is
    /// produced once and cannot be.
    fn try_clone(&self) -> Option<Self> {
        match &self.kind {
            BodyKind::Bytes(bytes) => Some(Self::from(bytes.clone())),
            BodyKind::Deferred(_) => None,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn into_native(self) -> reqwest::Body {
        match self.kind {
            BodyKind::Bytes(bytes) => reqwest::Body::from(bytes),
            BodyKind::Deferred(chunk) => {
                reqwest::Body::wrap_stream(futures_util::stream::once(chunk))
            }
        }
    }
}

impl From<Bytes> for Body {
    fn from(bytes: Bytes) -> Self {
        Self {
            kind: BodyKind::Bytes(bytes),
        }
    }
}

impl From<Vec<u8>> for Body {
    fn from(bytes: Vec<u8>) -> Self {
        Self::from(Bytes::from(bytes))
    }
}

impl From<String> for Body {
    fn from(text: String) -> Self {
        Self::from(Bytes::from(text))
    }
}

/// A body of one chunk that `chunk` produces when the transport first asks
/// for it: after the connection is open (and, to the Eidola backend,
/// attested). The durability barrier runs after `chunk` and before its bytes
/// are written, so a commit `chunk` makes is durable before the request
/// carries what it wrote.
pub(crate) fn deferred_body<F>(chunk: F) -> Body
where
    F: std::future::Future<Output = std::io::Result<Bytes>>
        + tinfoil_verifier::pipe::MaybeSend
        + 'static,
{
    Body {
        kind: BodyKind::Deferred(Box::pin(async move {
            let bytes = chunk.await?;
            crate::durability::barrier().await;
            Ok(bytes)
        })),
    }
}

/// A response: its head, and a body read as it arrives.
pub(crate) struct Response {
    kind: ResponseKind,
}

enum ResponseKind {
    #[cfg(not(target_family = "wasm"))]
    Native(reqwest::Response),
    Pipe(pipe::PipeResponse),
}

#[cfg(not(target_family = "wasm"))]
impl From<reqwest::Response> for Response {
    fn from(response: reqwest::Response) -> Self {
        Self {
            kind: ResponseKind::Native(response),
        }
    }
}

/// A response body's chunks, as they arrive: `Send` natively.
#[cfg(not(target_family = "wasm"))]
pub(crate) type BodyStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<Bytes, Error>> + Send>>;
/// A response body's chunks, as they arrive: `Send` natively.
#[cfg(target_family = "wasm")]
pub(crate) type BodyStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<Bytes, Error>>>>;

impl Response {
    fn pipe(response: pipe::PipeResponse) -> Self {
        Self {
            kind: ResponseKind::Pipe(response),
        }
    }

    pub(crate) fn status(&self) -> StatusCode {
        match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            ResponseKind::Native(response) => response.status(),
            ResponseKind::Pipe(response) => response.status,
        }
    }

    pub(crate) fn headers(&self) -> &HeaderMap {
        match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            ResponseKind::Native(response) => response.headers(),
            ResponseKind::Pipe(response) => &response.headers,
        }
    }

    /// The body's length as its head declares it — the peer's claim, for an
    /// early refusal, never for when to stop reading.
    pub(crate) fn content_length(&self) -> Option<u64> {
        match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            ResponseKind::Native(response) => response.content_length(),
            ResponseKind::Pipe(response) => response.content_length(),
        }
    }

    pub(crate) fn bytes_stream(self) -> BodyStream {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            ResponseKind::Native(response) => Box::pin(
                response
                    .bytes_stream()
                    .map(|chunk| chunk.map_err(Error::from)),
            ),
            ResponseKind::Pipe(response) => response.into_stream(),
        }
    }
}

/// A transport failure.
///
/// Its text never carries the request URL (a model link's query is its
/// authorization): [`crate::error::request_error_text`] renders a native one
/// without it, and the host-socket transport never writes it.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    #[cfg(not(target_family = "wasm"))]
    Native(reqwest::Error),
    /// The request could not be built.
    Builder(String),
    /// Sending it, or reading its answer, failed.
    Request(String),
}

#[cfg(not(target_family = "wasm"))]
impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self {
            kind: ErrorKind::Native(error),
        }
    }
}

impl Error {
    fn builder(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Builder(message.into()),
        }
    }

    fn request(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Request(message.into()),
        }
    }

    /// The request could not be built (a malformed URL, header or body).
    pub(crate) fn is_builder(&self) -> bool {
        match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            ErrorKind::Native(e) => e.is_builder(),
            ErrorKind::Builder(_) => true,
            ErrorKind::Request(_) => false,
        }
    }

    /// The error as text with no request URL in it, its causes appended.
    pub(crate) fn text_without_url(self) -> String {
        match self.kind {
            #[cfg(not(target_family = "wasm"))]
            ErrorKind::Native(e) => {
                let e = e.without_url();
                let mut chain = format!("{e}");
                let mut source = std::error::Error::source(&e);
                while let Some(err) = source {
                    use std::fmt::Write;
                    let _ = write!(chain, ": {err}");
                    source = err.source();
                }
                chain
            }
            ErrorKind::Builder(message) => format!("builder error: {message}"),
            ErrorKind::Request(message) => format!("error sending request: {message}"),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            #[cfg(not(target_family = "wasm"))]
            ErrorKind::Native(e) => e.fmt(f),
            ErrorKind::Builder(message) => write!(f, "builder error: {message}"),
            ErrorKind::Request(message) => write!(f, "error sending request: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl Default for Body {
    fn default() -> Self {
        Self::empty()
    }
}
