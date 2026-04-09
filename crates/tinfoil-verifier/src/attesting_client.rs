//! Per-handshake attestation via a hyper [`tower::Layer`] over reqwest's
//! connector.
//!
//! Every time the underlying connection pool needs a new TCP+TLS connection,
//! the wrapped connector finishes the TLS handshake and then performs an
//! inline HTTP/1.1 `GET /.well-known/tinfoil-attestation?v=3` over **the same
//! stream**. The response is verified before the connection is yielded back
//! to hyper for the real request. There is no cache: every new handshake is
//! re-attested. Subsequent HTTP requests on a pooled keepalive connection do
//! not re-trigger the connector and therefore do not re-attest, but they are
//! still bound to the same TLS key that was attested when the connection was
//! first established.
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
//! HTTP lifecycle for any connection it sees), so we frame the request and
//! parse the response ourselves. The wire format is fixed: one
//! request, one response, `Content-Length` or chunked transfer encoding.
//!
//! ## Important: this fixes freshness for *policy* but not for key compromise
//!
//! Re-attesting on every new TLS handshake means TCB-floor bumps and
//! `ALLOWED_MEASUREMENTS` changes take effect immediately rather than only at
//! process restart. It does **not** mitigate an attacker who has somehow
//! exfiltrated the enclave's long-lived TLS private key: the attestation
//! document is static (no nonce yet) and replayable as long as the attacker
//! can complete a TLS handshake with the bound key. Closing that gap requires
//! per-handshake nonces in `report_data`, which Tinfoil is adding upstream.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use http::Extensions;
use hyper_util::client::legacy::connect::Connection;
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::{Layer, Service};

use der::Decode;

use crate::measurement::EnclaveMeasurement;
use crate::{
    AtcFallback, Error, bundle, check_snp_measurement, check_tdx_measurement, sevsnp, sevsnp_crl,
    tdx,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Maximum wall-clock time the inline attestation exchange (write request,
/// read response, parse, verify chains, optional ATC backfill) is allowed to
/// take before the connector aborts the handshake. Bounds the worst case
/// where a backend completes the TLS handshake but stalls on the GET.
const ATTESTATION_DEADLINE: Duration = Duration::from_secs(10);

/// Build a `reqwest::Client` whose connector verifies enclave attestation on
/// every new TLS connection.
///
/// The returned client speaks HTTP/1.1 only (forced via ALPN) so that the
/// inline attestation request and any subsequent application requests share
/// a single connection lifecycle the connector layer can drive.
/// Owned, plumbing-friendly mirror of [`crate::AttestingClientConfig`] used
/// by [`build_attesting_client`]. We keep it crate-private so the public
/// config can stay borrow-flavored (`&'a [...]`) without forcing every
/// internal helper to carry a lifetime parameter.
pub(crate) struct BuildParams {
    pub inference_base_url: String,
    pub trusted_ark_der: Option<Vec<u8>>,
    pub trusted_ask_der: Option<Vec<u8>>,
    pub allowed_measurements: Vec<EnclaveMeasurement>,
    pub atc_fallback: AtcFallback,
    pub tdx_policy: tdx::TcbPolicy,
    pub tdx_observer: Option<tdx::TdxObserver>,
    pub snp_policy: sevsnp::SevSnpTcbPolicy,
    pub snp_observer: Option<sevsnp::SevSnpObserver>,
    pub tls_roots: Arc<rustls::RootCertStore>,
}

pub(crate) fn build_attesting_client(params: BuildParams) -> Result<reqwest::Client, Error> {
    let BuildParams {
        inference_base_url,
        trusted_ark_der,
        trusted_ask_der,
        allowed_measurements,
        atc_fallback,
        tdx_policy,
        tdx_observer,
        snp_policy,
        snp_observer,
        tls_roots,
    } = params;
    let host = crate::enclave_host(&inference_base_url);

    // Build a rustls config that pins ALPN to http/1.1 so the connection we
    // attest is the same connection hyper will use for the real request. The
    // root store comes from the caller verbatim — we do not consult any
    // bundled or OS source here. The server (running inside an enclave with
    // no system trust store) supplies `webpki-roots`; the CLI / macOS app
    // supply `rustls-native-certs` so developers can install local dev CAs
    // in their keychain. We deliberately do not inject the AMD attestation
    // ARK as a TLS root.
    let mut tls_config = rustls::ClientConfig::builder()
        .with_root_certificates((*tls_roots).clone())
        .with_no_client_auth();
    tls_config.alpn_protocols = vec![b"http/1.1".to_vec()];

    let check = Arc::new(AttestationCheck {
        allowed_measurements,
        attestation_path: "/.well-known/tinfoil-attestation?v=3".to_string(),
        attestation_host: host,
        trusted_ark_der,
        trusted_ask_der,
        atc_fallback,
        tdx_collateral_cache: tdx::CollateralCache::new(tls_roots.clone()),
        tdx_policy,
        tdx_observer,
        snp_policy,
        snp_observer,
        snp_crl_cache: sevsnp_crl::CrlCache::new(tls_roots),
    });

    reqwest::Client::builder()
        .use_preconfigured_tls(tls_config)
        .http1_only()
        .tls_info(true)
        .connect_timeout(Duration::from_secs(10))
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
                match tokio::time::timeout(ATTESTATION_DEADLINE, check.attest(conn)).await {
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

/// Per-client attestation policy and target.
struct AttestationCheck {
    allowed_measurements: Vec<EnclaveMeasurement>,
    attestation_path: String,
    attestation_host: String,
    trusted_ark_der: Option<Vec<u8>>,
    trusted_ask_der: Option<Vec<u8>>,
    /// ATC fallback target, used only when the self-contained v3 attestation
    /// document is missing required elements (today: the VCEK). The
    /// connector itself does not call AMD KDS — ATC is the single fallback.
    atc_fallback: AtcFallback,
    /// Per-client cache of TDX collateral fetched from Intel PCS, keyed by
    /// `(fmspc, ca)`. Bounds Intel PCS request volume to roughly one set of
    /// fetches per FMSPC per TCB advisory cycle. Unused on SEV-SNP backends.
    tdx_collateral_cache: tdx::CollateralCache,
    /// Acceptance policy applied on top of `dcap_verify`'s built-in checks.
    /// Unused on SEV-SNP backends.
    tdx_policy: tdx::TcbPolicy,
    /// Optional consumer-provided observer fired for every TDX
    /// attestation that completes signature verification (including ones
    /// the policy then rejects). Lets the consuming application record
    /// metrics or alerts without `tinfoil-verifier` taking a dependency
    /// on a metrics framework. Unused on SEV-SNP backends.
    tdx_observer: Option<tdx::TdxObserver>,
    /// Per-component minimum SVNs the SEV-SNP `reported_tcb` must satisfy.
    /// Also drives the rollback check (`reported_tcb >= committed_tcb`),
    /// which is structural and not configurable. Unused on TDX backends.
    snp_policy: sevsnp::SevSnpTcbPolicy,
    /// Optional consumer-provided observer fired for every SEV-SNP
    /// attestation that completes signature verification (including ones
    /// the policy then rejects). Same lifecycle and constraints as
    /// `tdx_observer`. Unused on TDX backends.
    snp_observer: Option<sevsnp::SevSnpObserver>,
    /// Per-client cache of AMD KDS CRLs keyed by AMD platform generation.
    /// Same stale-while-revalidate / single-flight semantics as the TDX
    /// collateral cache. Unused on TDX backends.
    snp_crl_cache: sevsnp_crl::CrlCache,
}

impl AttestationCheck {
    /// Run the attestation handshake on a freshly-handshaken connection and
    /// hand it back ready for the real request.
    async fn attest<C>(&self, conn: C) -> Result<C, Error>
    where
        C: Connection + hyper::rt::Read + hyper::rt::Write + Send + Sync + Unpin + 'static,
    {
        // Pull the peer certificate from reqwest's TLS info plumbing.
        let mut ext = Extensions::new();
        conn.connected().get_extras(&mut ext);
        let tls_info = ext.get::<reqwest::tls::TlsInfo>().ok_or_else(|| {
            Error::Connector(
                "no TLS info on freshly-handshaken connection (tls_info(true) \
                 must be set on the reqwest builder)"
                    .to_string(),
            )
        })?;
        let peer_cert_der = tls_info.peer_certificate().ok_or_else(|| {
            Error::Connector("peer certificate missing from TLS info".to_string())
        })?;
        let peer_spki = sevsnp::sha256_spki_from_der(peer_cert_der)?;

        // Wrap the hyper IO in TokioIo so we can use AsyncRead/Write extension
        // methods to drive a single inline HTTP/1.1 request without dropping
        // down past the response framing.
        let mut io = TokioIo::new(conn);

        let resolved = self.fetch_well_known(&mut io).await?;
        self.verify(&resolved, &peer_spki).await?;

        Ok(io.into_inner())
    }

    /// Issue an HTTP/1.1 GET against the configured v3 well-known endpoint
    /// and parse the body into a [`bundle::ResolvedAttestation`].
    async fn fetch_well_known<T>(
        &self,
        io: &mut TokioIo<T>,
    ) -> Result<bundle::ResolvedAttestation, Error>
    where
        T: hyper::rt::Read + hyper::rt::Write + Unpin,
    {
        let request = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {host}\r\n\
             Connection: keep-alive\r\n\
             Accept: application/json\r\n\
             User-Agent: tinfoil-verifier\r\n\
             \r\n",
            path = self.attestation_path,
            host = self.attestation_host,
        );
        io.write_all(request.as_bytes())
            .await
            .map_err(|e| Error::Connector(format!("write attestation request: {e}")))?;
        io.flush()
            .await
            .map_err(|e| Error::Connector(format!("flush attestation request: {e}")))?;

        let body = read_http1_response(io).await?;
        let doc: bundle::AttestationDocumentV3 = serde_json::from_slice(&body)
            .map_err(|e| Error::Connector(format!("v3 attestation JSON parse: {e}")))?;
        bundle::resolve_v3(&doc)
    }

    /// Verify a freshly-fetched attestation document against the peer cert
    /// the TLS handshake landed on.
    async fn verify(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
    ) -> Result<(), Error> {
        match resolved.platform {
            bundle::Platform::SevSnp => self.verify_snp(resolved, peer_spki).await,
            bundle::Platform::Tdx => self.verify_tdx(resolved, peer_spki).await,
        }
    }

    async fn verify_snp(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
    ) -> Result<(), Error> {
        let report = sevsnp::parse_report(&resolved.report_bytes)?;

        // VCEK source: prefer the self-contained document; otherwise fall back
        // to ATC. The connector never talks to AMD KDS — ATC is the single
        // fallback target. Once Tinfoil ships fully self-contained v3 reports
        // this fallback path will go cold.
        let vcek_der = match resolved.vcek_der.clone() {
            Some(v) => v,
            None => {
                tracing::debug!(
                    "well-known attestation document missing VCEK; consulting ATC fallback",
                );
                self.atc_fallback.fetch_vcek().await?
            }
        };

        // ARK/ASK come from exactly one of two trusted sources: the caller's
        // explicit `trusted_ark_der`/`trusted_ask_der` configuration (test
        // deployments using the tinfoil shim mock), or the built-in AMD Genoa
        // certs in the `sev` crate (production). We resolve to owned DER
        // bytes once so the same material can be used for chain
        // verification, CRL signature verification, and ASK/VCEK serial
        // extraction without re-loading the built-in certs three times.
        let (ark_der, ask_der) = sevsnp::resolve_chain_certs_der(
            self.trusted_ark_der.as_deref(),
            self.trusted_ask_der.as_deref(),
        )?;

        sevsnp::verify_report(&vcek_der, &report, Some(&ark_der), Some(&ask_der))?;

        // Check the AMD KDS CRL for revocation of either intermediate
        // (ASK) or leaf (VCEK). The chain verifies above by signature
        // alone — `sev` does not consult any CRL — so we layer this on
        // top here. The CRL is fetched directly from AMD KDS rather
        // than through Tinfoil's well-known doc on purpose: it is a
        // global signed object (no information leak about which chip
        // we are verifying) and a compromised operator could otherwise
        // serve a stale list to silently re-enable a revoked chip.
        //
        // The CRL is signed by AMD's real ARK, so this check only makes
        // sense in production where the chain is rooted in the built-in
        // Genoa ARK. Test deployments that supply a custom ARK (the
        // tinfoil shim mock) would always fail CRL signature verification
        // against their mock ARK and AMD KDS has no revocation entries
        // for mock chips anyway, so we skip the check entirely there.
        if self.trusted_ark_der.is_none() {
            let ark_x509 = x509_cert::Certificate::from_der(&ark_der)
                .map_err(|e| Error::CertParse(format!("failed to parse ARK DER: {e}")))?;
            let vcek_serial = sevsnp::cert_serial_from_der(&vcek_der)?;
            let ask_serial = sevsnp::cert_serial_from_der(&ask_der)?;
            self.snp_crl_cache
                .check_revocation(
                    sevsnp_crl::AmdGeneration::Genoa,
                    &ark_x509,
                    &[&vcek_serial, &ask_serial],
                )
                .await?;
        }

        // Apply the operator-configured TCB policy. The observer fires
        // *before* we propagate the policy result so consumers see the
        // full population of observed TCB levels — including ones we
        // will reject — for metrics and alerting.
        let (snp_observation, policy_result) = self.snp_policy.evaluate(&report);
        if let Some(observer) = &self.snp_observer {
            observer(&snp_observation);
        }
        policy_result?;

        let measurement_hex = hex::encode(report.measurement);
        let matched = check_snp_measurement(&self.allowed_measurements, &measurement_hex)?;

        if &report.report_data[..32] != peer_spki.as_slice() {
            return Err(Error::FingerprintMismatch {
                report_data: hex::encode(&report.report_data[..32]),
                enclave_cert: hex::encode(peer_spki),
            });
        }

        tracing::info!(
            measurement = %matched,
            tls_fingerprint = hex::encode(peer_spki),
            reported_tcb = %snp_observation.reported_tcb,
            committed_tcb = %snp_observation.committed_tcb,
            tcb_bucket = snp_observation.as_metric_label(),
            "SEV-SNP attestation verified for new connection",
        );
        Ok(())
    }

    async fn verify_tdx(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
    ) -> Result<(), Error> {
        let collateral = self
            .tdx_collateral_cache
            .get_or_fetch(&resolved.report_bytes)
            .await?;
        let result = tdx::verify_quote(
            &resolved.report_bytes,
            &collateral,
            &self.tdx_policy,
            self.tdx_observer.as_ref(),
        )?;

        let rtmr1_hex = hex::encode(result.rtmr1);
        let rtmr2_hex = hex::encode(result.rtmr2);
        let matched = check_tdx_measurement(&self.allowed_measurements, &rtmr1_hex, &rtmr2_hex)?;

        if &result.report_data[..32] != peer_spki.as_slice() {
            return Err(Error::FingerprintMismatch {
                report_data: hex::encode(&result.report_data[..32]),
                enclave_cert: hex::encode(peer_spki),
            });
        }

        tracing::info!(
            measurement = %matched,
            tls_fingerprint = hex::encode(peer_spki),
            "TDX attestation verified for new connection",
        );
        Ok(())
    }
}

/// Read a single HTTP/1.1 response from `io` and return its body bytes.
///
/// Supports `Content-Length` and `Transfer-Encoding: chunked`. Bounded so a
/// hostile endpoint cannot exhaust memory.
async fn read_http1_response<T>(io: &mut TokioIo<T>) -> Result<Vec<u8>, Error>
where
    T: hyper::rt::Read + hyper::rt::Write + Unpin,
{
    const MAX_HEAD: usize = 16 * 1024;
    const MAX_BODY: usize = 4 * 1024 * 1024;

    // Read until end of headers.
    let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    let head_len: usize;
    let status: u16;
    let content_length: Option<usize>;
    let chunked: bool;
    loop {
        let mut chunk = [0u8; 4096];
        let n = io
            .read(&mut chunk)
            .await
            .map_err(|e| Error::Connector(format!("read response head: {e}")))?;
        if n == 0 {
            return Err(Error::Connector(
                "EOF before HTTP response head".to_string(),
            ));
        }
        buf.extend_from_slice(&chunk[..n]);

        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut resp = httparse::Response::new(&mut headers);
        match resp
            .parse(&buf)
            .map_err(|e| Error::Connector(format!("httparse: {e}")))?
        {
            httparse::Status::Complete(parsed_len) => {
                head_len = parsed_len;
                status = resp.code.unwrap_or(0);
                let mut cl = None;
                let mut ch = false;
                for h in resp.headers.iter() {
                    if h.name.eq_ignore_ascii_case("content-length") {
                        cl = std::str::from_utf8(h.value)
                            .ok()
                            .and_then(|s| s.trim().parse().ok());
                    } else if h.name.eq_ignore_ascii_case("transfer-encoding")
                        && std::str::from_utf8(h.value)
                            .map(|s| {
                                s.split(',')
                                    .any(|p| p.trim().eq_ignore_ascii_case("chunked"))
                            })
                            .unwrap_or(false)
                    {
                        ch = true;
                    }
                }
                content_length = cl;
                chunked = ch;
                break;
            }
            httparse::Status::Partial => {
                if buf.len() > MAX_HEAD {
                    return Err(Error::Connector("HTTP response head too large".to_string()));
                }
                continue;
            }
        }
    }

    if !(200..300).contains(&status) {
        return Err(Error::Connector(format!(
            "attestation endpoint returned HTTP {status}"
        )));
    }

    // Body bytes already buffered after the head.
    if chunked {
        let mut body = Vec::new();
        let mut rest = buf.split_off(head_len);
        loop {
            // Find chunk-size CRLF.
            let crlf = loop {
                if let Some(pos) = rest.windows(2).position(|w| w == b"\r\n") {
                    break pos;
                }
                if rest.len() > 1024 {
                    return Err(Error::Connector("chunk size line too long".to_string()));
                }
                let mut chunk = [0u8; 256];
                let n = io
                    .read(&mut chunk)
                    .await
                    .map_err(|e| Error::Connector(format!("read chunk size line: {e}")))?;
                if n == 0 {
                    return Err(Error::Connector("EOF in chunk size line".to_string()));
                }
                rest.extend_from_slice(&chunk[..n]);
            };
            let size_line = std::str::from_utf8(&rest[..crlf])
                .map_err(|e| Error::Connector(format!("chunk size utf8: {e}")))?;
            let size_str = size_line.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_str, 16)
                .map_err(|e| Error::Connector(format!("chunk size parse: {e}")))?;
            rest.drain(..crlf + 2);

            if size == 0 {
                // Last chunk; consume the trailing CRLF (assume no trailers).
                while rest.len() < 2 {
                    let mut chunk = [0u8; 64];
                    let n = io
                        .read(&mut chunk)
                        .await
                        .map_err(|e| Error::Connector(format!("read final chunk CRLF: {e}")))?;
                    if n == 0 {
                        return Err(Error::Connector("EOF after final chunk".to_string()));
                    }
                    rest.extend_from_slice(&chunk[..n]);
                }
                break;
            }

            if body.len() + size > MAX_BODY {
                return Err(Error::Connector("chunked body too large".to_string()));
            }
            while rest.len() < size + 2 {
                let mut chunk = vec![0u8; 4096];
                let n = io
                    .read(&mut chunk)
                    .await
                    .map_err(|e| Error::Connector(format!("read chunk body: {e}")))?;
                if n == 0 {
                    return Err(Error::Connector("EOF inside chunk body".to_string()));
                }
                rest.extend_from_slice(&chunk[..n]);
            }
            body.extend_from_slice(&rest[..size]);
            rest.drain(..size + 2);
        }
        Ok(body)
    } else if let Some(len) = content_length {
        if len > MAX_BODY {
            return Err(Error::Connector("Content-Length exceeds limit".to_string()));
        }
        let need = head_len + len;
        while buf.len() < need {
            let remaining = need - buf.len();
            let mut chunk = vec![0u8; remaining.min(8192)];
            let n = io
                .read(&mut chunk)
                .await
                .map_err(|e| Error::Connector(format!("read response body: {e}")))?;
            if n == 0 {
                return Err(Error::Connector(
                    "EOF inside Content-Length body".to_string(),
                ));
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        if buf.len() > need {
            return Err(Error::Connector(
                "attestation endpoint sent extra bytes after Content-Length body".to_string(),
            ));
        }
        Ok(buf[head_len..need].to_vec())
    } else {
        Err(Error::Connector(
            "attestation response missing Content-Length and not chunked".to_string(),
        ))
    }
}
