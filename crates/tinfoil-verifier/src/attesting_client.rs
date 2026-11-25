//! Per-handshake attestation via a hyper [`tower::Layer`] over reqwest's
//! connector.
//!
//! Every time the underlying connection pool needs a new TCP+TLS connection,
//! the wrapped connector finishes the TLS handshake, then hands the stream to
//! [`AttestationCheck::attest_stream`] — the one implementation of what is
//! checked, which the byte-pipe transport ([`crate::pipe`]) drives too. The
//! connection is yielded back to hyper for the real request only once the
//! check passes. There is no cache: every new handshake is re-attested with a
//! new nonce. Subsequent HTTP requests on a pooled keepalive connection do not
//! re-trigger the connector and therefore do not re-attest, but they are still
//! bound to the same TLS key that was attested (and the same nonce-bound
//! report) when the connection was first established.
//!
//! ## Why inline HTTP/1.1?
//!
//! The connector layer can intercept connections post-handshake but pre-HTTP,
//! which is the only place we can guarantee that the attestation document
//! comes from the *exact* backend the data plane will subsequently talk to —
//! critical when the upstream sits behind a load balancer that may otherwise
//! route a side-channel attestation fetch to a different instance.
//!
//! Once the inner connection is in our hands we cannot ask hyper's high-level
//! `Client` to drive a request on it (the high-level client owns the entire
//! HTTP lifecycle for any connection it sees), so the check frames the request
//! and parses the response itself (`attest::read_http1_response`).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use http::Extensions;
use hyper_util::client::legacy::connect::Connection;
use hyper_util::rt::TokioIo;
use tower::{Layer, Service};

use crate::Error;
use crate::attest::{ATTESTATION_DEADLINE, AttestationCheck};
use crate::sevsnp_crl;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// How long an attested connection may sit with nothing to send before TCP
/// starts probing whether its peer is still there, and the interval between
/// probes; [`TCP_KEEPALIVE_PROBES`] unanswered probes close it.
///
/// A connection waiting on a long answer (a blocking completion, which sends
/// nothing until it is done) or idle in the pool carries no bytes of its own,
/// so a peer that vanished without closing it (a host gone, a path cut) is
/// noticed only this way: within a minute (15 s, then three probes 15 s
/// apart), as a connection error rather than a wait to the caller's own
/// deadline. Shorter costs a probe every few seconds on every idle pooled
/// connection; longer keeps a dead connection, and what waits on it, longer.
/// The values are `reqwest`'s defaults, set here so the client does not
/// depend on them.
const TCP_KEEPALIVE: Duration = Duration::from_secs(15);

/// See [`TCP_KEEPALIVE`].
const TCP_KEEPALIVE_PROBES: u32 = 3;

/// Build a `reqwest::Client` whose connector verifies enclave attestation on
/// every new TLS connection.
///
/// The returned client speaks HTTP/1.1 only (forced via ALPN) so that the
/// inline attestation request and any subsequent application requests share
/// a single connection lifecycle the connector layer can drive.
pub(crate) fn build_attesting_client(
    tls_config: rustls::ClientConfig,
    check: Arc<AttestationCheck>,
) -> Result<reqwest::Client, Error> {
    // **No redirects.** reqwest's default follows up to ten and replays a
    // cloneable body on `307`/`308`. Attestation alone does not pin the
    // origin: a redirect opens a fresh connection that is attested in its own
    // right, so a cross-origin `Location` is refused only when the new peer
    // cannot prove an allowed measurement over its own TLS key (every plain
    // `http://` target, every non-enclave host) — another enclave running a
    // pinned measurement under a different hostname would pass, and receive
    // the body. Refusing every redirect keeps a request at the origin it was
    // made to and turns the rest into the upstream's own `3xx` rather than an
    // attestation failure. No same-origin exception: the Eidola server has no
    // redirecting route, so a `3xx` from it is an error answer either way.
    reqwest::Client::builder()
        .use_preconfigured_tls(tls_config)
        .redirect(reqwest::redirect::Policy::none())
        .http1_only()
        .tls_info(true)
        .connect_timeout(Duration::from_secs(10))
        .tcp_keepalive(TCP_KEEPALIVE)
        .tcp_keepalive_interval(TCP_KEEPALIVE)
        .tcp_keepalive_retries(TCP_KEEPALIVE_PROBES)
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(usize::MAX)
        .connector_layer(AttestingConnectorLayer { check })
        .build()
        .map_err(|e| Error::Tls(format!("failed to build attesting client: {e}")))
}

/// Tower layer that wraps reqwest's inner connector with attestation
/// verification.
#[derive(Clone)]
struct AttestingConnectorLayer {
    check: Arc<AttestationCheck>,
}

impl<S> Layer<S> for AttestingConnectorLayer {
    type Service = AttestingConnectorService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        AttestingConnectorService {
            inner,
            check: self.check.clone(),
        }
    }
}

#[derive(Clone)]
struct AttestingConnectorService<S> {
    inner: S,
    check: Arc<AttestationCheck>,
}

impl<S, Req> Service<Req> for AttestingConnectorService<S>
where
    S: Service<Req> + Clone + Send + Sync + 'static,
    S::Future: Send + 'static,
    S::Error: Into<BoxError> + Send + 'static,
    S::Response: Connection + hyper::rt::Read + hyper::rt::Write + Send + Sync + Unpin + 'static,
    Req: Send + 'static,
{
    type Response = S::Response;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, BoxError>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        // Standard tower pattern: swap the inner service we just polled into
        // the future, leaving a fresh clone behind for the next poll/call.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let check = self.check.clone();
        Box::pin(async move {
            let conn = inner.call(req).await.map_err(Into::into)?;

            // Bound the inline attestation exchange so a stalled upstream
            // can't wedge a hyper pool slot indefinitely.
            let attested =
                match tokio::time::timeout(ATTESTATION_DEADLINE, attest(&check, conn)).await {
                    Ok(result) => result?,
                    Err(_) => {
                        return Err(Box::new(Error::AttestationTimeout {
                            seconds: ATTESTATION_DEADLINE.as_secs(),
                        }) as BoxError);
                    }
                };
            Ok(attested)
        })
    }
}

/// Attest a freshly-handshaken reqwest connection and hand it back ready for
/// the real request: the peer certificate from reqwest's TLS info, the clock
/// from the system, the checks from [`AttestationCheck::attest_stream`].
async fn attest<C>(check: &AttestationCheck, conn: C) -> Result<C, Error>
where
    C: Connection + hyper::rt::Read + hyper::rt::Write + Send + Sync + Unpin + 'static,
{
    let mut ext = Extensions::new();
    conn.connected().get_extras(&mut ext);
    let tls_info = ext.get::<reqwest::tls::TlsInfo>().ok_or_else(|| {
        Error::Connector(
            "no TLS info on freshly-handshaken connection (tls_info(true) \
             must be set on the reqwest builder)"
                .to_string(),
        )
    })?;
    let peer_cert_der = tls_info
        .peer_certificate()
        .ok_or_else(|| Error::Connector("peer certificate missing from TLS info".to_string()))?
        .to_vec();

    // TokioIo turns hyper's IO into tokio's AsyncRead/AsyncWrite, which is
    // what the check drives its inline request over.
    let mut io = TokioIo::new(conn);
    check
        .attest_stream(&mut io, &peer_cert_der, sevsnp_crl::unix_now()?)
        .await?;
    Ok(io.into_inner())
}
