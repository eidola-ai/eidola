//! Tinfoil attestation verification with per-handshake attesting.
//!
//! Verifies that every new TLS connection to a Tinfoil inference enclave
//! terminates inside genuine AMD SEV-SNP or Intel TDX hardware running an
//! allowed code measurement.
//!
//! All verification happens in the data-plane connector layer (see
//! [`attesting_client`]). On every new TCP+TLS handshake the connector
//! issues an inline HTTP/1.1 attestation request over the *same* stream that
//! will subsequently carry application traffic, then verifies the response
//! binds to the peer's TLS public key before yielding the connection back to
//! hyper. There is no fingerprint cache, so policy changes (TCB floor,
//! allowed measurements) take effect on the next connection without a
//! process restart, and there is no separate startup bootstrap path — the
//! first real request through the returned client is also the first
//! attestation. Callers that want fail-fast-at-startup semantics can issue
//! a single trivial request through the client themselves.
//!
//! The enclave's self-contained
//! `/.well-known/tinfoil-attestation?v=3` document is the source of truth.
//! Tinfoil's ATC service is consulted only as a fallback for elements that
//! a non-self-contained v3 document is missing (today: the VCEK certificate,
//! until upstream ships fully self-contained reports). The verifier never
//! talks to AMD KDS for the chain itself — ATC is the single fallback target —
//! though it does fetch AMD KDS CRLs in production mode for revocation checks.
//!
//! SEV-SNP verification is delegated to the [`sev`](https://crates.io/crates/sev)
//! crate. TDX Quote V4 verification is delegated to [`dcap_qvl`].
//!
//! # Prerequisites
//!
//! A rustls `CryptoProvider` must be installed before calling [`attesting_client`].
//! The eidola server does this in `main.rs` via `rustls_rustcrypto::provider()`.
//!
//! # TLS root sourcing
//!
//! `tinfoil-verifier` is intentionally agnostic about where TLS trust roots
//! come from. Callers populate [`AttestingClientConfig::tls_roots`] themselves
//! and the same store is used for the attested inference endpoint, ATC
//! fallback lookups, and AMD KDS CRL fetches. Each consumer picks the source
//! that fits its environment:
//!
//! - **Server (in enclave):** `webpki-roots`. The server runs `FROM scratch`
//!   inside an enclave with no system trust store, so it bundles the Mozilla
//!   list. Tinfoil's production cert and the public services it talks to all
//!   chain under it.
//! - **CLI / macOS app:** `rustls-native-certs`. Picks up the developer's OS
//!   keychain so locally-installed dev CAs (e.g. the tinfoil shim mock's
//!   `tls-ca.pem`) work without recompilation.
//!
//! This crate deliberately does **not** depend on either source so neither
//! gets dragged into the wrong consumer transitively.

mod attesting_client;
pub mod bundle;
mod error;
pub mod measurement;
pub mod sevsnp;
pub mod sevsnp_crl;
pub mod tdx;

pub use bundle::Platform;
pub use error::Error;
pub use measurement::{EnclaveMeasurement, MatchedMeasurement, TdxMeasurement};
pub use sevsnp::{SevSnpObserver, SevSnpTcbObservation, SevSnpTcbPolicy, SevSnpTcbSvns};
pub use tdx::{TcbPolicy as TdxTcbPolicy, TdxObserver, TdxTcbObservation, TdxTcbStatus};

/// Configuration for [`attesting_client`].
pub struct AttestingClientConfig<'a> {
    /// Allowed enclave releases. Each entry pairs a SEV-SNP measurement with a
    /// TDX measurement; the verifier picks the matching field based on the
    /// platform observed in the attestation document.
    pub allowed_measurements: &'a [EnclaveMeasurement],
    /// Base URL of the inference endpoint (e.g. `https://inference.tinfoil.sh/v1`).
    /// The `/.well-known/tinfoil-attestation` endpoint is derived from the origin.
    pub inference_base_url: &'a str,
    /// TLS root store used for **all** outbound HTTPS performed by the
    /// resulting client: the attested inference endpoint, ATC fallback
    /// lookups, and AMD KDS CRL fetches. The verifier is intentionally
    /// agnostic about where these roots come from — the caller decides
    /// whether to populate the store from `webpki-roots`, the OS keychain
    /// via `rustls-native-certs`, a custom PEM, or some union. The server
    /// (running inside an enclave with no system trust store) typically
    /// uses `webpki-roots`; the CLI and macOS app use `rustls-native-certs`
    /// so developers can install local dev CAs in their keychain. Custom
    /// SEV-SNP attestation roots (`trusted_ark_der` / `trusted_ask_der`)
    /// are deliberately *not* added here; they only feed the SEV-SNP chain
    /// verifier.
    pub tls_roots: rustls::RootCertStore,
    /// Optional ATC URL override for **fallback** lookups when the enclave's
    /// own v3 well-known document is missing pieces (today: the VCEK).
    /// Defaults to [`bundle::DEFAULT_ATC_URL`] when `None`. ATC is never
    /// consulted when the well-known document is self-contained.
    pub atc_url: Option<&'a str>,
    /// Source repository to attest against (e.g.
    /// `tinfoilsh/confidential-model-router`). Used as the `repo` field in
    /// the ATC `POST /attestation` request body when an ATC fallback lookup
    /// is required. When `None`, the verifier will fail any handshake whose
    /// well-known document is not self-contained, since there is no
    /// fallback target to consult.
    pub enclave_repo: Option<&'a str>,
    /// Optional custom trusted ARK (Root CA) DER bytes. Overrides the
    /// built-in AMD Genoa ARK in the SEV-SNP attestation chain verifier.
    /// **Not** added to any TLS root store; if you need TLS to trust a
    /// custom CA (e.g. for the tinfoil shim mock), install the cert in your
    /// system trust store.
    pub trusted_ark_der: Option<&'a [u8]>,
    /// Optional custom trusted ASK DER bytes. Same caveats as
    /// [`Self::trusted_ark_der`].
    pub trusted_ask_der: Option<&'a [u8]>,
    /// Optional allowlist of Intel advisory IDs (e.g. `INTEL-SA-00837`)
    /// the operator has explicitly reviewed and accepted.
    ///
    /// When `None` or empty, TDX attestation follows Intel's recommended
    /// verifier policy: `UpToDate` accepted silently; `*Needed` levels
    /// accepted with a warning; `OutOfDate*` and `Revoked` rejected.
    /// When non-empty, an `OutOfDate*` level is also accepted (with a
    /// warning) iff every advisory ID associated with the matched TCB
    /// level is contained in the allowlist. Unrelated to SEV-SNP, which
    /// has its own minimum-firmware floor in
    /// [`crate::sevsnp::verify_tcb_policy`].
    pub tdx_advisory_allowlist: Option<&'a [&'a str]>,
    /// Optional observer fired for every TDX attestation that completes
    /// signature verification, **including ones the policy rejects**.
    /// Lets the consuming application record metrics, traces, or alerts
    /// without `tinfoil-verifier` taking a dependency on a metrics
    /// framework.
    ///
    /// The callback runs synchronously inside the connector layer on the
    /// TLS handshake hot path, so it must be cheap and non-blocking
    /// (e.g. an OTel counter increment is fine; HTTP I/O is not).
    /// Unused on SEV-SNP backends.
    pub tdx_observer: Option<TdxObserver>,
    /// Operator-supplied minimum TCB SVNs the SEV-SNP `reported_tcb`
    /// must satisfy. When `None`, defaults to
    /// [`SevSnpTcbPolicy::amd_recommended`] (`bootloader >= 0x07`,
    /// `snp >= 0x0E`, `microcode >= 0x48`, no `tee` floor). The rollback
    /// check (`reported_tcb >= committed_tcb`) is structural and always
    /// applied regardless of this setting. Unrelated to TDX, which has
    /// its own [`Self::tdx_advisory_allowlist`].
    pub snp_min_tcb: Option<SevSnpTcbPolicy>,
    /// Optional observer fired for every SEV-SNP attestation that
    /// completes signature verification, **including ones the policy
    /// rejects**. Same lifecycle and constraints as
    /// [`Self::tdx_observer`]. Unused on TDX backends.
    pub snp_observer: Option<SevSnpObserver>,
}

/// Build a `reqwest::Client` whose connector verifies enclave attestation on
/// every new TLS connection.
///
/// The client is ready to use immediately and performs no network I/O during
/// construction. The first request through it will trigger the connector,
/// which:
///
/// 1. Completes the TCP+TLS handshake.
/// 2. Issues an inline HTTP/1.1 `GET /.well-known/tinfoil-attestation?v=3`
///    over the *same* connection.
/// 3. Falls back to ATC for any element the v3 document is missing
///    (currently the VCEK).
/// 4. Verifies the AMD VCEK chain, the report signature, the TCB floor, the
///    measurement against `allowed_measurements`, and that the report's
///    `report_data[0..32]` matches `sha256(SPKI(peer_cert))`.
/// 5. Yields the connection to hyper for the real request.
///
/// Callers that want fail-fast-at-startup semantics should make one trivial
/// request (e.g. `client.get(format!("{base}/v1/models")).send().await`)
/// after construction and treat its outcome as the readiness check.
pub async fn attesting_client(config: AttestingClientConfig<'_>) -> Result<reqwest::Client, Error> {
    let host = enclave_host(config.inference_base_url);
    let tls_roots = std::sync::Arc::new(config.tls_roots);
    let atc_fallback = AtcFallback {
        url: config.atc_url.map(str::to_string),
        repo: config.enclave_repo.map(str::to_string),
        enclave_host: host,
        tls_roots: tls_roots.clone(),
    };

    let tdx_policy = match config.tdx_advisory_allowlist {
        Some(list) if !list.is_empty() => {
            tdx::TcbPolicy::with_advisory_allowlist(list.iter().copied())
        }
        _ => tdx::TcbPolicy::intel_recommended(),
    };

    let snp_policy = config.snp_min_tcb.unwrap_or_default();

    attesting_client::build_attesting_client(attesting_client::BuildParams {
        inference_base_url: config.inference_base_url.to_string(),
        trusted_ark_der: config.trusted_ark_der.map(|d| d.to_vec()),
        trusted_ask_der: config.trusted_ask_der.map(|d| d.to_vec()),
        allowed_measurements: config.allowed_measurements.to_vec(),
        atc_fallback,
        tdx_policy,
        tdx_observer: config.tdx_observer,
        snp_policy,
        snp_observer: config.snp_observer,
        tls_roots,
    })
}

/// Configuration for the ATC fallback path used by the per-handshake
/// connector when a self-contained attestation document is missing required
/// elements.
#[derive(Clone)]
pub(crate) struct AtcFallback {
    pub url: Option<String>,
    pub repo: Option<String>,
    pub enclave_host: String,
    /// Shared TLS root store used to validate the ATC endpoint's cert.
    pub tls_roots: std::sync::Arc<rustls::RootCertStore>,
}

impl AtcFallback {
    /// Fetch a bundle from ATC and return the VCEK certificate from it.
    pub async fn fetch_vcek(&self) -> Result<Vec<u8>, Error> {
        let repo = self.repo.as_deref().ok_or_else(|| {
            Error::Bundle(
                "well-known attestation document is not self-contained and no \
                 enclave_repo is configured for ATC fallback"
                    .to_string(),
            )
        })?;
        let display_url = self.url.as_deref().unwrap_or(bundle::DEFAULT_ATC_URL);
        tracing::info!(
            "Fetching ATC fallback VCEK from {display_url} (enclave={}, repo={repo})",
            self.enclave_host
        );
        let bundle = bundle::fetch_bundle(
            self.url.as_deref(),
            &self.enclave_host,
            repo,
            &self.tls_roots,
        )
        .await?;
        sevsnp::decode_base64(&bundle.vcek)
    }
}

/// Check a SEV-SNP measurement against the allowed list (case-insensitive).
/// Returns the matched [`MatchedMeasurement::SevSnp`] on success.
pub(crate) fn check_snp_measurement(
    allowed: &[EnclaveMeasurement],
    measurement_hex: &str,
) -> Result<MatchedMeasurement, Error> {
    let hit = allowed
        .iter()
        .find(|m| m.snp_measurement.eq_ignore_ascii_case(measurement_hex));
    match hit {
        Some(m) => Ok(MatchedMeasurement::SevSnp(m.snp_measurement.clone())),
        None => Err(Error::MeasurementMismatch {
            observed: MatchedMeasurement::SevSnp(measurement_hex.to_string()),
            allowed_count: allowed.len(),
        }),
    }
}

/// Check a TDX RTMR1+RTMR2 pair against the allowed list (case-insensitive).
/// Returns the matched [`MatchedMeasurement::Tdx`] on success.
pub(crate) fn check_tdx_measurement(
    allowed: &[EnclaveMeasurement],
    rtmr1_hex: &str,
    rtmr2_hex: &str,
) -> Result<MatchedMeasurement, Error> {
    let hit = allowed.iter().find(|m| {
        m.tdx_measurement.rtmr1.eq_ignore_ascii_case(rtmr1_hex)
            && m.tdx_measurement.rtmr2.eq_ignore_ascii_case(rtmr2_hex)
    });
    match hit {
        Some(m) => Ok(MatchedMeasurement::Tdx(m.tdx_measurement.clone())),
        None => Err(Error::MeasurementMismatch {
            observed: MatchedMeasurement::Tdx(TdxMeasurement {
                rtmr1: rtmr1_hex.to_string(),
                rtmr2: rtmr2_hex.to_string(),
            }),
            allowed_count: allowed.len(),
        }),
    }
}

/// Extract the bare host (no scheme, no port, no path) from an inference base URL.
///
/// Used as the `enclaveUrl` parameter in the ATC `POST /attestation` request,
/// and as the `Host` header in the per-handshake inline attestation request.
pub(crate) fn enclave_host(inference_base_url: &str) -> String {
    let after_scheme = match inference_base_url.find("://") {
        Some(scheme_end) => &inference_base_url[scheme_end + 3..],
        None => inference_base_url,
    };
    let authority_end = after_scheme.find('/').unwrap_or(after_scheme.len());
    let authority = &after_scheme[..authority_end];
    // Strip an optional port
    match authority.rfind(':') {
        Some(colon) => authority[..colon].to_string(),
        None => authority.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enclave_host_strips_scheme_path_port() {
        assert_eq!(
            enclave_host("https://inference.tinfoil.sh/v1"),
            "inference.tinfoil.sh"
        );
        assert_eq!(
            enclave_host("https://inference.tinfoil.sh"),
            "inference.tinfoil.sh"
        );
        assert_eq!(
            enclave_host("https://inference.tinfoil.sh:8443/v1"),
            "inference.tinfoil.sh"
        );
        assert_eq!(enclave_host("inference.tinfoil.sh"), "inference.tinfoil.sh");
    }
}
