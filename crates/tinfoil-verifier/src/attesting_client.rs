//! Per-handshake attestation via a hyper [`tower::Layer`] over reqwest's
//! connector.
//!
//! Every time the underlying connection pool needs a new TCP+TLS connection,
//! the wrapped connector finishes the TLS handshake, generates a fresh random
//! nonce, and then performs an inline HTTP/1.1
//! `GET /.well-known/tinfoil-attestation?nonce=<hex>` over **the same
//! stream**. The response is verified before the connection is yielded back
//! to hyper for the real request. There is no cache: every new handshake is
//! re-attested with a new nonce. Subsequent HTTP requests on a pooled
//! keepalive connection do not re-trigger the connector and therefore do not
//! re-attest, but they are still bound to the same TLS key that was attested
//! (and the same nonce-bound report) when the connection was first
//! established.
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
//! ## Freshness — and its limits
//!
//! Re-attesting on every new TLS handshake means a change to the TCB floor
//! or to the pinned measurements takes effect immediately rather than only
//! at process restart. The per-handshake nonce adds *freshness*: the enclave
//! folds our random nonce into the hardware report's `REPORT_DATA`
//! (`REPORT_DATA == sha256(report-data-v1 URI || nonce ||
//! sha256(crypto_material) || sha256(device_evidence))`), and the endorsed
//! crypto section carries the SPKI hash of the cert we handshook with. A
//! stale or captured document therefore cannot be replayed against a
//! different nonce.
//!
//! It does **not** close the key-exfiltration gap. The report binds the
//! long-term TLS *key*, not the live TLS *session*, so an attacker who has
//! exfiltrated that key can still actively MITM the connection — reading the
//! plaintext, since holding the signing key lets them drive the ECDHE
//! regardless of TLS 1.3 forward secrecy — and relay a fresh nonce-bound
//! report fetched from the enclave's public attestation endpoint (which serves
//! any nonce). Every check here still passes. Closing this requires channel
//! binding (committing a TLS-session value into `report_data`); today it rests
//! on the TLS key staying sealed inside the enclave.

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
use sha2::Digest as _;

use crate::bundle::Platform;
use crate::measurement::{
    CompiledPin, CompiledPlatform, CompiledSevSnpPin, CompiledTdxPin, MatchedMeasurement,
    SevSnpLaunchMeasurement, TdxLaunchMeasurement,
};
use crate::{Error, bundle, device, sevsnp, sevsnp_crl, tdx};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Maximum wall-clock time the inline attestation exchange (write request,
/// read response, parse, and verify chains) is allowed to
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
    pub allowed_measurements: Vec<CompiledPin>,
    pub snp_policy: sevsnp::SevSnpTcbPolicy,
    pub snp_observer: Option<sevsnp::SevSnpObserver>,
    pub attestation_observer: Option<crate::AttestationObserver>,
    pub tls_roots: Arc<rustls::RootCertStore>,
}

pub(crate) fn build_attesting_client(params: BuildParams) -> Result<reqwest::Client, Error> {
    let BuildParams {
        inference_base_url,
        trusted_ark_der,
        trusted_ask_der,
        allowed_measurements,
        snp_policy,
        snp_observer,
        attestation_observer,
        tls_roots,
    } = params;
    let host = crate::enclave_host(&inference_base_url);

    // Build a rustls config that pins ALPN to http/1.1 so the connection we
    // attest is the same connection hyper will use for the real request. The
    // root store comes from the caller verbatim — we do not consult any
    // bundled or OS source here. The gateway (running inside an enclave with
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
        intel_root_der: tdx::INTEL_SGX_ROOT_CA_DER.to_vec(),
        attestation_path: "/.well-known/tinfoil-attestation".to_string(),
        attestation_host: host,
        trusted_ark_der,
        trusted_ask_der,
        snp_policy,
        snp_observer,
        attestation_observer,
    });

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
    /// Platform-tagged pins. Evidence is compared only with entries of its
    /// own platform.
    allowed_measurements: Vec<CompiledPin>,
    /// The Intel SGX root every TDX quote's chains must end in. Always the
    /// built-in root outside this module's tests.
    intel_root_der: Vec<u8>,
    /// Base well-known path; a fresh `?nonce=<hex>` query is appended per
    /// handshake.
    attestation_path: String,
    attestation_host: String,
    trusted_ark_der: Option<Vec<u8>>,
    trusted_ask_der: Option<Vec<u8>>,
    /// Per-component minimum SVNs the SEV-SNP `reported_tcb` must satisfy.
    /// Also drives the rollback check (`reported_tcb >= committed_tcb`),
    /// which is structural and not configurable. Unused on TDX backends.
    snp_policy: sevsnp::SevSnpTcbPolicy,
    /// Optional consumer-provided observer fired for every SEV-SNP
    /// attestation that completes signature verification (including ones
    /// the policy then rejects). Unused on TDX backends.
    snp_observer: Option<sevsnp::SevSnpObserver>,
    /// Optional consumer-provided observer fired for every successful
    /// attestation. Same lifecycle as the platform-specific observers.
    attestation_observer: Option<crate::AttestationObserver>,
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

        // Fresh random nonce per handshake. The enclave binds it into the
        // hardware report's REPORT_DATA, so a captured document cannot be
        // replayed against a different nonce. It does not bind the live TLS
        // session or make an exfiltrated TLS key safe.
        let nonce = bundle::random_nonce()?;

        // Wrap the hyper IO in TokioIo so we can use AsyncRead/Write extension
        // methods to drive a single inline HTTP/1.1 request without dropping
        // down past the response framing.
        let mut io = TokioIo::new(conn);

        let resolved = self.fetch_well_known(&mut io, &nonce).await?;
        self.verify(&resolved, &peer_spki, &nonce)?;

        Ok(io.into_inner())
    }

    /// Issue an HTTP/1.1 GET for a fresh, nonce-bound attestation document over
    /// the same connection and parse it into a [`bundle::ResolvedAttestation`].
    async fn fetch_well_known<T>(
        &self,
        io: &mut TokioIo<T>,
        nonce: &[u8; bundle::NONCE_LEN],
    ) -> Result<bundle::ResolvedAttestation, Error>
    where
        T: hyper::rt::Read + hyper::rt::Write + Unpin,
    {
        let request = format!(
            "GET {path}?nonce={nonce} HTTP/1.1\r\n\
             Host: {host}\r\n\
             Connection: keep-alive\r\n\
             Accept: application/json\r\n\
             User-Agent: tinfoil-verifier\r\n\
             \r\n",
            path = self.attestation_path,
            nonce = hex::encode(nonce),
            host = self.attestation_host,
        );
        io.write_all(request.as_bytes())
            .await
            .map_err(|e| Error::Connector(format!("write attestation request: {e}")))?;
        io.flush()
            .await
            .map_err(|e| Error::Connector(format!("flush attestation request: {e}")))?;

        let body = read_http1_response(io).await?;
        self.parse_document(&body)
    }

    /// Parse a fetched document, refusing a platform with no pinned entry
    /// before its payloads are decoded. `verify` re-checks the platform on
    /// the resolved document.
    fn parse_document(&self, body: &[u8]) -> Result<bundle::ResolvedAttestation, Error> {
        let pinned: Vec<Platform> = [Platform::SevSnp, Platform::Tdx]
            .into_iter()
            .filter(|platform| {
                self.allowed_measurements
                    .iter()
                    .any(|pin| pin.platform() == *platform)
            })
            .collect();
        bundle::parse_document_for(body, &pinned)
    }

    /// Verify a freshly-fetched attestation document against the peer cert the
    /// TLS handshake landed on and the nonce we just sent.
    fn verify(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
        sent_nonce: &[u8; bundle::NONCE_LEN],
    ) -> Result<(), Error> {
        // These envelope checks are not authentication by themselves. The
        // platform path must prove the signed report carries the same
        // recomputed REPORT_DATA before the document is trusted.
        self.verify_document_binding(resolved, peer_spki, sent_nonce)?;

        // The document's author chooses the platform branch, so only a
        // platform the caller pinned is a branch at all. With no entry for
        // it, the evidence is refused before it is even authenticated: a
        // SEV-SNP-only pin set refuses TDX no matter how genuine the quote.
        let candidates: Vec<&CompiledPin> = self
            .allowed_measurements
            .iter()
            .filter(|pin| pin.platform() == resolved.platform)
            .collect();
        if candidates.is_empty() {
            return Err(Error::PlatformNotPinned {
                platform: resolved.platform,
            });
        }

        match resolved.platform {
            Platform::SevSnp => self.verify_snp(resolved, peer_spki, &candidates),
            Platform::Tdx => {
                let now = sevsnp_crl::unix_now()?;
                self.verify_tdx(resolved, peer_spki, &candidates, now)
            }
        }
    }

    /// Platform-independent checks binding the fresh document to the TLS key
    /// presented on this connection and to this nonce (not to the TLS session):
    ///
    /// 1. The echoed nonce equals the one we sent (freshness / anti-replay).
    /// 2. The endorsed `tls` key equals `sha256(SPKI(peer_cert))`.
    fn verify_document_binding(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
        sent_nonce: &[u8; bundle::NONCE_LEN],
    ) -> Result<(), Error> {
        if resolved.nonce != *sent_nonce {
            return Err(Error::NonceMismatch {
                sent: hex::encode(sent_nonce),
                echoed: hex::encode(resolved.nonce),
            });
        }

        if &resolved.tls_key_fp != peer_spki {
            return Err(Error::FingerprintMismatch {
                attested: hex::encode(resolved.tls_key_fp),
                peer: hex::encode(peer_spki),
            });
        }
        Ok(())
    }

    /// Cross-check that the hardware report's `REPORT_DATA` equals the SHA-256
    /// of the document's `report_data` inputs. This is what binds the
    /// attacker-uncontrollable hardware report to the (nonce, TLS key, HPKE
    /// key) the document claims — every one of those claimed fields is thereby
    /// authenticated by the AMD signature over the report.
    fn verify_report_data_binding(
        &self,
        resolved: &bundle::ResolvedAttestation,
        report_data: &[u8; 64],
    ) -> Result<(), Error> {
        if &resolved.report_data != report_data {
            return Err(Error::ReportDataMismatch {
                expected: hex::encode(resolved.report_data),
                observed: hex::encode(report_data),
            });
        }
        Ok(())
    }

    fn verify_snp(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
        candidates: &[&CompiledPin],
    ) -> Result<(), Error> {
        let report = sevsnp::parse_report(&resolved.report_bytes)?;

        // Structural field checks the launch measurement does not cover —
        // most importantly that the guest policy forbids DEBUG (a relaunch
        // of the pinned image with DEBUG=1 keeps the same measurement but
        // lets the hypervisor decrypt guest memory).
        sevsnp::check_report_hygiene(&report)?;

        let vcek_der = resolved.vcek_der.as_deref().ok_or_else(|| {
            Error::Bundle("SEV-SNP document carries no VCEK collateral".to_string())
        })?;
        let crl_der = resolved.crl_der.as_deref().ok_or_else(|| {
            Error::Bundle("SEV-SNP document carries no AMD CRL collateral".to_string())
        })?;
        let carried_ask_der = resolved.ask_der.as_deref().ok_or_else(|| {
            Error::Bundle("SEV-SNP document carries no ASK certificate".to_string())
        })?;
        let carried_ark_der = resolved.ark_der.as_deref().ok_or_else(|| {
            Error::Bundle("SEV-SNP document carries no ARK certificate".to_string())
        })?;

        // The document carries ASK + ARK so verification is network-free, but
        // they remain untrusted transport. Require byte identity with the
        // caller's explicit mock chain or the built-in AMD Genoa chain before
        // using them for VCEK/report and CRL verification.
        let (trusted_ark_der, trusted_ask_der) = sevsnp::resolve_chain_certs_der(
            self.trusted_ark_der.as_deref(),
            self.trusted_ask_der.as_deref(),
        )?;
        if carried_ark_der != trusted_ark_der {
            return Err(Error::CertChain(
                "document-carried ARK does not match the trusted AMD root".to_string(),
            ));
        }
        if carried_ask_der != trusted_ask_der {
            return Err(Error::CertChain(
                "document-carried ASK does not match the trusted AMD intermediate".to_string(),
            ));
        }

        sevsnp::verify_report(
            vcek_der,
            &report,
            Some(carried_ark_der),
            Some(carried_ask_der),
        )?;

        // Collateral is untrusted transport. Require a complete, direct CRL,
        // verify its ARK identity and signature, enforce its signed half-open
        // validity window, and reject either AMD-issued chain certificate if
        // listed. This is the only revocation source: no AMD KDS request is
        // made during verification.
        let ark_x509 = x509_cert::Certificate::from_der(carried_ark_der)
            .map_err(|e| Error::CertParse(format!("failed to parse ARK DER: {e}")))?;
        let vcek_serial = sevsnp::cert_serial_from_der(vcek_der)?;
        let ask_serial = sevsnp::cert_serial_from_der(carried_ask_der)?;
        sevsnp_crl::check_revocation(crl_der, &ark_x509, &[&vcek_serial, &ask_serial])?;

        // Apply the operator-configured TCB policy. The observer fires
        // *before* we propagate the policy result so consumers see the
        // full population of observed TCB levels — including ones we
        // will reject — for metrics and alerting.
        let (snp_observation, policy_result) = self.snp_policy.evaluate(&report);
        if let Some(observer) = &self.snp_observer {
            observer(&snp_observation);
        }
        policy_result?;

        let policy = sevsnp::raw_policy(&resolved.report_bytes)?;
        let (observed, matching) =
            appraise_snp_launch(candidates, &report.measurement, &report.host_data, policy)?;
        let pcr_digest = format!("{}:{}", observed.measurement, observed.host_data);
        let matched = MatchedMeasurement::SevSnp(observed);

        // Bind the hardware report to the nonce/TLS-key/HPKE-key the document
        // claimed. `verify_document_binding` already proved `tls_key_fp ==
        // peer_spki` and the nonce is fresh; this proves the AMD-signed report
        // actually commits to those same values — and to the exact device
        // evidence section the next check reads.
        self.verify_report_data_binding(resolved, &report.report_data)?;
        first_passing(matching, |(pin, snp)| {
            sevsnp::check_guest_policy(policy, snp.guest_policy)?;
            check_device_policy(resolved, pin)
        })?;

        tracing::info!(
            measurement = %matched,
            tls_fingerprint = hex::encode(peer_spki),
            reported_tcb = %snp_observation.reported_tcb,
            committed_tcb = %snp_observation.committed_tcb,
            tcb_bucket = snp_observation.as_metric_label(),
            "SEV-SNP attestation verified for new connection",
        );

        if let Some(observer) = &self.attestation_observer {
            observer(crate::VerifiedAttestation {
                platform: Platform::SevSnp,
                matched_measurement: matched,
                attestation_hash: hex::encode(sha2::Sha256::digest(&resolved.report_bytes)),
                attestation_doc: resolved.report_bytes.clone(),
                pcr_digest,
                peer_spki_hash: hex::encode(peer_spki),
            });
        }

        Ok(())
    }

    /// Authenticate a TDX quote with the document's captured Intel
    /// collateral at `now`, then appraise it against the pinned entries.
    fn verify_tdx(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
        candidates: &[&CompiledPin],
        now: u64,
    ) -> Result<(), Error> {
        let responses = resolved.intel_pcs.as_deref().ok_or_else(|| {
            Error::Bundle("TDX document carries no Intel PCS collateral".to_string())
        })?;
        let quote =
            tdx::authenticate(&resolved.report_bytes, responses, &self.intel_root_der, now)?;
        self.verify_tdx_authenticated(resolved, peer_spki, candidates, &quote)
    }

    /// Appraise an authenticated TDX quote. Entries are selected by launch
    /// identity (MRTD + MRCONFIGID); among those, the first whose machine
    /// policy and device requirement both hold is the match.
    fn verify_tdx_authenticated(
        &self,
        resolved: &bundle::ResolvedAttestation,
        peer_spki: &[u8; 32],
        candidates: &[&CompiledPin],
        quote: &tdx::AuthenticatedQuote,
    ) -> Result<(), Error> {
        let observed = TdxLaunchMeasurement {
            mrtd: hex::encode(quote.report.mr_td),
            mrconfigid: hex::encode(quote.report.mr_config_id),
        };
        let matching: Vec<(&CompiledPin, &CompiledTdxPin)> = candidates
            .iter()
            .filter_map(|pin| match &pin.platform {
                CompiledPlatform::Tdx(tdx_pin)
                    if tdx_pin.mrtd == quote.report.mr_td
                        && tdx_pin.mrconfigid == quote.report.mr_config_id =>
                {
                    Some((*pin, tdx_pin.as_ref()))
                }
                _ => None,
            })
            .collect();
        if matching.is_empty() {
            return Err(Error::MeasurementMismatch {
                observed: MatchedMeasurement::Tdx(observed),
                allowed_count: candidates.len(),
            });
        }

        // Machine policy first, so a policy violation is reported as such
        // rather than as a REPORT_DATA or device mismatch.
        first_passing(matching.clone(), |(_, tdx_pin)| {
            tdx::appraise(quote, tdx_pin)
        })?;
        self.verify_report_data_binding(resolved, &quote.report.report_data)?;
        // The device requirement is read only now that REPORT_DATA has
        // authenticated the device evidence section.
        first_passing(matching, |(pin, tdx_pin)| {
            tdx::appraise(quote, tdx_pin)?;
            check_device_policy(resolved, pin)
        })?;

        let pcr_digest = format!("{}:{}", observed.mrtd, observed.mrconfigid);
        let matched = MatchedMeasurement::Tdx(observed);
        tracing::info!(
            measurement = %matched,
            tls_fingerprint = hex::encode(peer_spki),
            mr_seam = hex::encode(quote.report.mr_seam),
            tee_tcb_svn = hex::encode(quote.report.tee_tcb_svn),
            fmspc = hex::encode(quote.fmspc),
            ppid = hex::encode(quote.ppid),
            tcb_info_evaluation_data_number = quote.tcb_info_evaluation_data_number,
            qe_identity_evaluation_data_number = quote.qe_identity_evaluation_data_number,
            "TDX attestation verified for new connection",
        );

        if let Some(observer) = &self.attestation_observer {
            observer(crate::VerifiedAttestation {
                platform: Platform::Tdx,
                matched_measurement: matched,
                attestation_hash: hex::encode(sha2::Sha256::digest(&resolved.report_bytes)),
                attestation_doc: resolved.report_bytes.clone(),
                pcr_digest,
                peer_spki_hash: hex::encode(peer_spki),
            });
        }
        Ok(())
    }
}

/// The SEV-SNP entries an authenticated report's launch matches.
///
/// Entries are selected by launch identity: the image's digest and the
/// deployment's config hash (HOST_DATA). The digest alone names every
/// deployment of the image, anyone's. Among those, the guest policy, which
/// the host chose at launch and which is outside the digest, must satisfy at
/// least one entry's (`sevsnp::check_guest_policy`); the entries are returned
/// for the device check, which re-applies it per entry.
#[allow(clippy::type_complexity)]
fn appraise_snp_launch<'a>(
    candidates: &[&'a CompiledPin],
    measurement: &[u8; 48],
    host_data: &[u8; 32],
    policy: u64,
) -> Result<
    (
        SevSnpLaunchMeasurement,
        Vec<(&'a CompiledPin, &'a CompiledSevSnpPin)>,
    ),
    Error,
> {
    let observed = SevSnpLaunchMeasurement {
        measurement: hex::encode(measurement),
        host_data: hex::encode(host_data),
    };
    let matching: Vec<(&CompiledPin, &CompiledSevSnpPin)> = candidates
        .iter()
        .filter_map(|pin| match &pin.platform {
            CompiledPlatform::SevSnp(snp)
                if snp.measurement == *measurement && snp.host_data == *host_data =>
            {
                Some((*pin, snp))
            }
            _ => None,
        })
        .collect();
    if matching.is_empty() {
        return Err(Error::MeasurementMismatch {
            observed: MatchedMeasurement::SevSnp(observed),
            allowed_count: candidates.len(),
        });
    }
    first_passing(matching.clone(), |(_, snp)| {
        sevsnp::check_guest_policy(policy, snp.guest_policy)
    })?;
    Ok((observed, matching))
}

/// Apply a matched entry's device-evidence requirement. Only called after
/// the signed `REPORT_DATA` has been shown to commit to the device section.
fn check_device_policy(
    resolved: &bundle::ResolvedAttestation,
    pin: &CompiledPin,
) -> Result<(), Error> {
    match pin.expected_gpus {
        Some(expected) => {
            device::check_gpu_evidence(&resolved.device_evidence, expected, &resolved.nonce)
        }
        None => Ok(()),
    }
}

/// The first candidate `check` accepts; otherwise the first candidate's
/// error, which names the closest pin's violation.
fn first_passing<T>(
    candidates: Vec<T>,
    mut check: impl FnMut(T) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut first_error = None;
    for candidate in candidates {
        match check(candidate) {
            Ok(()) => return Ok(()),
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    Err(first_error
        .unwrap_or_else(|| Error::Connector("no candidate measurement to check".to_string())))
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

            // Never `body.len() + size`: `size` is the peer's, and the
            // release profile builds without overflow checks.
            if size > MAX_BODY - body.len() {
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::measurement::{CompiledPin, compile_pins};
    use crate::tdx::tests::{
        FIXTURE_MRTD, FIXTURE_NOW, FIXTURE_QUOTE, authenticated, fixture_responses,
    };
    use crate::{AllowedMeasurement, TdxPin};

    const PEER_SPKI: [u8; 32] = [0x19; 32];
    const NONCE: [u8; 32] = [0x47; 32];

    fn check(pins: &[AllowedMeasurement]) -> AttestationCheck {
        AttestationCheck {
            allowed_measurements: compile_pins(pins).unwrap(),
            intel_root_der: tdx::INTEL_SGX_ROOT_CA_DER.to_vec(),
            attestation_path: "/.well-known/tinfoil-attestation".to_string(),
            attestation_host: "enclave.example".to_string(),
            trusted_ark_der: None,
            trusted_ask_der: None,
            snp_policy: sevsnp::SevSnpTcbPolicy::default(),
            snp_observer: None,
            attestation_observer: None,
        }
    }

    fn resolved(platform: Platform, report_bytes: Vec<u8>) -> bundle::ResolvedAttestation {
        bundle::ResolvedAttestation {
            platform,
            report_bytes,
            report_data: [0x5a; 64],
            nonce: NONCE,
            tls_key_fp: PEER_SPKI,
            hpke_key: [0x45; 32],
            vcek_der: None,
            ask_der: None,
            ark_der: None,
            crl_der: None,
            intel_pcs: Some(fixture_responses()),
            device_evidence: vec![],
        }
    }

    /// The server pin a client builds from its release record: SEV-SNP only.
    fn server_pin() -> AllowedMeasurement {
        AllowedMeasurement::sev_snp(crate::measurement::tests::sev_snp_pin())
    }

    /// The platform-branch attack: a genuine TDX quote presented to a
    /// client whose pins are SEV-SNP only, as the server pin is. The
    /// document chooses the TDX branch; with no TDX entry there is no such
    /// branch, and the quote is never even parsed.
    #[test]
    fn tdx_evidence_against_sev_snp_only_pins_is_refused_before_authentication() {
        let check = check(&[server_pin()]);

        for report in [FIXTURE_QUOTE.to_vec(), b"not a quote".to_vec()] {
            let mut doc = resolved(Platform::Tdx, report);
            let err = check.verify(&doc, &PEER_SPKI, &NONCE).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::PlatformNotPinned {
                        platform: Platform::Tdx
                    }
                ),
                "{err}"
            );
            doc.intel_pcs = None;
            let err = check.verify(&doc, &PEER_SPKI, &NONCE).unwrap_err();
            assert!(matches!(err, Error::PlatformNotPinned { .. }), "{err}");
        }
    }

    /// The handshake parser is gated on the client's own pins: an SEV-SNP
    /// client handed a TDX envelope it could not even decode answers
    /// `PlatformNotPinned`.
    #[test]
    fn handshake_parser_refuses_unpinned_platforms_before_decoding() {
        let tdx_envelope = serde_json::to_vec(&serde_json::json!({
            "format": bundle::ATTESTATION_V3_FORMAT,
            "challenge": {"nonce": "", "report_data": "", "report_data_algorithm": ""},
            "cpu_evidence": {
                "format": bundle::TDX_QUOTE_V1_FORMAT,
                "report_base64": "%%%",
                "endorsed": {"crypto_material_hash": "", "device_evidence_hash": ""},
            },
            "crypto_material": "",
            "device_evidence": "",
            "collateral": [],
        }))
        .unwrap();
        let snp_only = check(&[server_pin()]);
        let err = snp_only.parse_document(&tdx_envelope).err().unwrap();
        assert!(
            matches!(
                err,
                Error::PlatformNotPinned {
                    platform: Platform::Tdx
                }
            ),
            "{err}"
        );
        let both = check(&[
            server_pin(),
            AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin()),
        ]);
        let err = both.parse_document(&tdx_envelope).err().unwrap();
        assert!(matches!(err, Error::Bundle(_)), "{err}");
    }

    #[test]
    fn sev_snp_evidence_against_tdx_only_pins_is_refused() {
        let check = check(&[AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin())]);
        let err = check
            .verify(
                &resolved(Platform::SevSnp, vec![0; 1184]),
                &PEER_SPKI,
                &NONCE,
            )
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::PlatformNotPinned {
                    platform: Platform::SevSnp
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn document_binding_runs_before_platform_dispatch() {
        let check = check(&[server_pin()]);
        let err = check
            .verify(&resolved(Platform::Tdx, vec![]), &PEER_SPKI, &[0; 32])
            .unwrap_err();
        assert!(matches!(err, Error::NonceMismatch { .. }), "{err}");
        let err = check
            .verify(&resolved(Platform::Tdx, vec![]), &[0; 32], &NONCE)
            .unwrap_err();
        assert!(matches!(err, Error::FingerprintMismatch { .. }), "{err}");
    }

    fn snp_candidates(pins: &[AllowedMeasurement]) -> Vec<CompiledPin> {
        compile_pins(pins).unwrap()
    }

    /// The adversarial relaunches of a pinned SEV-SNP image: the image's
    /// digest is fixed, so the host's levers are the config it passes
    /// (HOST_DATA) and the guest policy. Each is refused; the honest launch
    /// is accepted.
    #[test]
    fn sev_snp_launches_are_selected_by_digest_and_config_then_held_to_the_policy() {
        let pin = crate::measurement::tests::sev_snp_pin();
        let compiled = snp_candidates(&[AllowedMeasurement::sev_snp(pin.clone())]);
        let candidates: Vec<&CompiledPin> = compiled.iter().collect();
        let measurement = [0xab; 48];
        let host_data: [u8; 32] = hex::decode(&pin.host_data).unwrap().try_into().unwrap();

        let (observed, matching) =
            appraise_snp_launch(&candidates, &measurement, &host_data, 0x30133).unwrap();
        assert_eq!(matching.len(), 1);
        assert_eq!(observed.host_data, pin.host_data);

        // The pinned image launched with another config: anyone's deployment
        // of the same release, or ours with a changed secret hash or shape.
        let other_config: [u8; 32] = hex::decode(crate::SevSnpPin::host_data_for_config(b"other"))
            .unwrap()
            .try_into()
            .unwrap();
        for host_data in [other_config, [0; 32]] {
            let err =
                appraise_snp_launch(&candidates, &measurement, &host_data, 0x30133).unwrap_err();
            assert!(
                matches!(&err, Error::MeasurementMismatch { observed: MatchedMeasurement::SevSnp(s), .. }
                    if s.host_data == hex::encode(host_data)),
                "{err}"
            );
        }
        // Another image with our config.
        let err = appraise_snp_launch(&candidates, &[0xcd; 48], &host_data, 0x30133).unwrap_err();
        assert!(matches!(err, Error::MeasurementMismatch { .. }), "{err}");
        // The pinned image and config with a weaker policy: DEBUG (the
        // hypervisor may decrypt guest memory), a migration agent, CXL
        // memory, or an older firmware ABI.
        for policy in [0xb0133, 0x70133, 0x230133, 0x30100] {
            let err =
                appraise_snp_launch(&candidates, &measurement, &host_data, policy).unwrap_err();
            assert!(matches!(err, Error::Report(_)), "{policy:#x}: {err}");
        }
        // A TDX entry never matches SEV-SNP evidence, whatever it pins.
        let tdx = snp_candidates(&[AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin())]);
        let tdx: Vec<&CompiledPin> = tdx.iter().collect();
        assert!(appraise_snp_launch(&tdx, &measurement, &host_data, 0x30133).is_err());
    }

    #[test]
    fn sev_snp_entries_sharing_a_launch_identity_are_alternatives_by_policy() {
        let strict = crate::measurement::tests::sev_snp_pin();
        let mut newer = strict.clone();
        newer.guest_policy = "0x30155".into();
        let compiled = snp_candidates(&[
            AllowedMeasurement::sev_snp(newer),
            AllowedMeasurement::sev_snp(strict.clone()),
        ]);
        let candidates: Vec<&CompiledPin> = compiled.iter().collect();
        let host_data: [u8; 32] = hex::decode(&strict.host_data).unwrap().try_into().unwrap();
        appraise_snp_launch(&candidates, &[0xab; 48], &host_data, 0x30133)
            .expect("the second entry's policy holds");
        let err = appraise_snp_launch(&candidates, &[0xab; 48], &host_data, 0x30100).unwrap_err();
        // The first entry's violation is the one reported.
        assert!(err.to_string().contains("0x30155"), "{err}");
    }

    fn fixture_pin(mrtd: &str) -> AllowedMeasurement {
        let mut pin = crate::measurement::tests::tdx_pin();
        pin.mrtd = mrtd.to_string();
        pin.mrconfigid = "00".repeat(48);
        AllowedMeasurement::tdx(pin)
    }

    /// The real quote through the handshake verifier: it authenticates
    /// offline, selects the entry by MRTD + MRCONFIGID, and is refused by
    /// the IGVM-model appraisal because its launch extended RTMR0.
    #[test]
    fn real_tdx_quote_is_authenticated_then_appraised() {
        let doc = resolved(Platform::Tdx, FIXTURE_QUOTE.to_vec());

        let pinned = check(&[fixture_pin(FIXTURE_MRTD)]);
        let candidates: Vec<&CompiledPin> = pinned.allowed_measurements.iter().collect();
        let err = pinned
            .verify_tdx(&doc, &PEER_SPKI, &candidates, FIXTURE_NOW)
            .unwrap_err();
        assert!(
            matches!(&err, Error::TdxPolicy(m) if m.contains("RTMR0")),
            "{err}"
        );

        let other = check(&[fixture_pin(&"11".repeat(48))]);
        let candidates: Vec<&CompiledPin> = other.allowed_measurements.iter().collect();
        let err = other
            .verify_tdx(&doc, &PEER_SPKI, &candidates, FIXTURE_NOW)
            .unwrap_err();
        assert!(
            matches!(&err, Error::MeasurementMismatch { observed: MatchedMeasurement::Tdx(t), .. }
                if t.mrtd == FIXTURE_MRTD),
            "{err}"
        );

        let mut tampered = doc;
        tampered.intel_pcs.as_mut().unwrap().pop();
        let err = pinned
            .verify_tdx(&tampered, &PEER_SPKI, &candidates, FIXTURE_NOW)
            .unwrap_err();
        assert!(matches!(err, Error::Quote(_)), "{err}");
    }

    fn accept(
        check: &AttestationCheck,
        doc: &bundle::ResolvedAttestation,
        quote: &tdx::AuthenticatedQuote,
    ) -> Result<(), Error> {
        let candidates: Vec<&CompiledPin> = check.allowed_measurements.iter().collect();
        check.verify_tdx_authenticated(doc, &PEER_SPKI, &candidates, quote)
    }

    fn bound_quote(doc: &bundle::ResolvedAttestation) -> tdx::AuthenticatedQuote {
        let mut quote = authenticated();
        quote.report.report_data = doc.report_data;
        quote
    }

    #[test]
    fn authenticated_quote_matching_its_pin_and_report_data_is_accepted() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut check = check(&[AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin())]);
        check.allowed_measurements[0].expected_gpus = None;
        let sink = seen.clone();
        check.attestation_observer = Some(Arc::new(move |att: crate::VerifiedAttestation| {
            sink.lock().unwrap().push(att);
        }));
        let doc = resolved(Platform::Tdx, b"quote".to_vec());
        accept(&check, &doc, &bound_quote(&doc)).expect("accepted");

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].platform, Platform::Tdx);
        let pin = crate::measurement::tests::tdx_pin();
        assert_eq!(
            seen[0].pcr_digest,
            format!("{}:{}", pin.mrtd, pin.mrconfigid)
        );
        assert!(
            matches!(&seen[0].matched_measurement, MatchedMeasurement::Tdx(t) if t.mrtd == pin.mrtd)
        );
    }

    #[test]
    fn report_data_not_committing_to_this_document_is_refused() {
        let check = check(&[AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin())]);
        let doc = resolved(Platform::Tdx, b"quote".to_vec());
        let mut quote = bound_quote(&doc);
        quote.report.report_data[63] ^= 1;
        let err = accept(&check, &doc, &quote).unwrap_err();
        assert!(matches!(err, Error::ReportDataMismatch { .. }), "{err}");
    }

    #[test]
    fn the_matched_entrys_gpu_requirement_is_enforced() {
        let pin = crate::measurement::tests::tdx_pin();
        let check = check(&[AllowedMeasurement::tdx(pin).with_expected_gpus(2)]);
        let mut doc = resolved(Platform::Tdx, b"quote".to_vec());
        let quote = bound_quote(&doc);

        doc.device_evidence = vec![crate::device::tests::gpu("gpu0", &NONCE)];
        let err = accept(&check, &doc, &quote).unwrap_err();
        assert!(matches!(err, Error::DeviceEvidence(_)), "{err}");

        doc.device_evidence
            .push(crate::device::tests::gpu("gpu1", &[0; 32]));
        let err = accept(&check, &doc, &quote).unwrap_err();
        assert!(matches!(err, Error::DeviceEvidence(_)), "{err}");

        doc.device_evidence[1] = crate::device::tests::gpu("gpu1", &NONCE);
        accept(&check, &doc, &quote).expect("two nonce-bound GPUs");
    }

    /// Entries sharing a launch identity are alternatives: the first whose
    /// machine policy and device requirement both hold is the match.
    #[test]
    fn entries_sharing_a_launch_identity_are_alternatives() {
        let mut other_module = crate::measurement::tests::tdx_pin();
        other_module.policy.mr_seam = vec!["33".repeat(48)];
        let doc = resolved(Platform::Tdx, b"quote".to_vec());
        let quote = bound_quote(&doc);

        let check_a = check(&[
            AllowedMeasurement::tdx(other_module.clone()),
            AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin()),
        ]);
        accept(&check_a, &doc, &quote).expect("second entry's MR_SEAM matches");

        let only_other = check(&[AllowedMeasurement::tdx(other_module)]);
        let err = accept(&only_other, &doc, &quote).unwrap_err();
        assert!(
            matches!(&err, Error::TdxPolicy(m) if m.contains("MR_SEAM")),
            "{err}"
        );

        let gpu_or_cpu = check(&[
            AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin()).with_expected_gpus(8),
            AllowedMeasurement::tdx(crate::measurement::tests::tdx_pin()).with_expected_gpus(0),
        ]);
        accept(&gpu_or_cpu, &doc, &quote).expect("the zero-GPU entry matches");
    }

    #[test]
    fn mrconfigid_selects_the_entry_for_this_deployments_config() {
        let mut other_config = crate::measurement::tests::tdx_pin();
        other_config.mrconfigid = TdxPin::mrconfigid_for_config(b"another config");
        let check = check(&[AllowedMeasurement::tdx(other_config)]);
        let doc = resolved(Platform::Tdx, b"quote".to_vec());
        let err = accept(&check, &doc, &bound_quote(&doc)).unwrap_err();
        assert!(matches!(err, Error::MeasurementMismatch { .. }), "{err}");
    }

    /// Run `read_http1_response` over an in-memory pipe whose far end has
    /// already written `response` and hung up.
    async fn read_response(response: &[u8]) -> Result<Vec<u8>, Error> {
        let (near, mut far) = tokio::io::duplex(64 * 1024);
        far.write_all(response).await.unwrap();
        drop(far);
        let mut io = TokioIo::new(TokioIo::new(near));
        read_http1_response(&mut io).await
    }

    #[tokio::test]
    async fn a_chunk_size_past_the_body_bound_is_refused_not_wrapped() {
        // The peer is unauthenticated here: this response is read before
        // the document it carries is verified. A one-byte chunk, then a
        // chunk claiming `usize::MAX` bytes — `1 + usize::MAX` wraps to 0
        // when overflow checks are off (the release profile), slipping
        // under the bound and panicking on the slice that follows.
        let response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
                         1\r\nA\r\nffffffffffffffff\r\nB";
        let err = read_response(response).await.unwrap_err();
        assert!(
            err.to_string().contains("chunked body too large"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn a_chunked_body_is_reassembled() {
        let response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
                         3\r\nabc\r\n2;ext=1\r\nde\r\n0\r\n\r\n";
        assert_eq!(read_response(response).await.unwrap(), b"abcde");
    }
}
