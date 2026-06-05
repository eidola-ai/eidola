//! Tinfoil attestation verification with per-handshake attesting.
//!
//! Verifies that every new TLS connection to a Tinfoil inference enclave
//! terminates inside genuine AMD SEV-SNP or Intel TDX hardware running an
//! allowed code measurement.
//!
//! All verification happens in the data-plane connector layer (see
//! [`attesting_client`]). On every new TCP+TLS handshake the connector
//! generates a fresh random nonce and issues an inline HTTP/1.1 attestation
//! request (`?nonce=<hex>`) over the *same* stream that will subsequently
//! carry application traffic, then verifies that the enclave's *freshly
//! collected* hardware report commits to that exact nonce and to the peer's
//! TLS public key before yielding the connection back to hyper. There is no
//! fingerprint cache, so policy changes (TCB floor, allowed measurements)
//! take effect on the next connection without a process restart, and there is
//! no separate startup bootstrap path — the first real request through the
//! returned client is also the first attestation. Callers that want
//! fail-fast-at-startup semantics can issue a single trivial request through
//! the client themselves.
//!
//! Because the report's `REPORT_DATA` binds the per-handshake nonce, a stale
//! or captured attestation document can't be replayed against a fresh nonce:
//! the verifier knows a live, genuine CC machine produced this report *now*.
//!
//! The nonce alone binds the enclave's long-term TLS *key* (the cert SPKI),
//! not the live TLS *session*, which would leave a gap: an attacker holding an
//! exfiltrated TLS key could actively MITM the connection and relay a fresh
//! nonce-bound report from the enclave's public endpoint. To close that, the
//! verifier also binds the **TLS channel**: the enclave folds the RFC 9266
//! `tls-exporter` of the session it terminates into `REPORT_DATA`, and the
//! verifier checks it equals *this* session's exporter (obtained from the
//! reqwest connection). A relaying MITM terminates a different TLS session
//! than the one it relays to the enclave, so the two exporters differ and the
//! check fails. This holds even against a stolen TLS key, since producing a
//! report bound to *our* session's exporter requires the genuine hardware
//! terminating *our* session. The check is enforced whenever the enclave
//! provides a binding; [`AttestingClientConfig::require_channel_binding`]
//! additionally rejects enclaves that provide none (for once the upstream
//! enclave reliably emits it — current Tinfoil enclaves predate the binding,
//! so it defaults off and the nonce alone still guarantees freshness).
//!
//! The enclave also signs the whole document with its TLS leaf key
//! (ECDSA — P-384 in production, P-256 for the shim mock); the verifier checks
//! that signature against the certificate carried in the document, which must
//! in turn match the peer cert the handshake landed on.
//!
//! The fresh `/.well-known/tinfoil-attestation?nonce=<hex>` document is the
//! source of truth. It is *not* fully self-contained — Tinfoil builds it
//! without the VCEK certificate — so the verifier consults Tinfoil's ATC
//! service as a fallback to backfill the VCEK. The verifier never talks to
//! AMD KDS for the chain itself — ATC is the single fallback target — though
//! it does fetch AMD KDS CRLs in production mode for revocation checks.
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

/// Details of a verified TEE attestation, emitted after each successful
/// new-connection attestation check.
#[derive(Debug, Clone)]
pub struct VerifiedAttestation {
    /// TEE platform (SEV-SNP or TDX).
    pub platform: Platform,
    /// The enclave measurement that matched the allowed list.
    pub matched_measurement: MatchedMeasurement,
    /// SHA-256 of the raw attestation report bytes (hex-encoded).
    pub attestation_hash: String,
    /// Raw attestation report bytes (SEV-SNP report or TDX quote).
    pub attestation_doc: Vec<u8>,
    /// Platform-specific code measurement digest (hex-encoded).
    /// For SEV-SNP: the 48-byte launch digest.
    /// For TDX: `{rtmr1}:{rtmr2}` (two 48-byte digests, colon-separated).
    pub pcr_digest: String,
    /// SHA-256 of the peer TLS certificate's SPKI (hex-encoded).
    pub peer_spki_hash: String,
}

/// Callback fired after each successful new-connection attestation.
/// Runs synchronously on the TLS handshake hot path — must be cheap
/// and non-blocking (e.g. push to a vec or increment a counter).
pub type AttestationObserver = std::sync::Arc<dyn Fn(VerifiedAttestation) + Send + Sync>;

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
    /// Optional observer fired on every successful attestation verification
    /// for a new TLS connection. Receives the full attestation details
    /// including the raw report bytes, matched measurement, and TLS
    /// binding hash. Same lifecycle and constraints as [`Self::tdx_observer`].
    pub attestation_observer: Option<AttestationObserver>,
    /// Require the enclave to bind the TLS session via an RFC 9266
    /// `tls-exporter` channel binding.
    ///
    /// When the enclave binds the channel, the verifier *always* checks that
    /// the bound exporter equals this session's exporter — this is what
    /// defeats a man-in-the-middle holding an exfiltrated TLS key (their
    /// client-facing session has a different exporter than the one they relay
    /// to the genuine enclave). This flag controls only what happens when the
    /// enclave provides **no** binding:
    ///
    /// - `false` (default): accept it. The per-handshake nonce still
    ///   guarantees freshness. Use during rollout, before the upstream enclave
    ///   is known to emit the binding.
    /// - `true`: reject it. Use once the upstream enclave reliably binds the
    ///   channel, to prevent a downgrade.
    ///
    /// Independent of platform; the binding rides in `report_data` and is
    /// authenticated by the hardware report.
    pub require_channel_binding: bool,
}

/// Build a `reqwest::Client` whose connector verifies enclave attestation on
/// every new TLS connection.
///
/// The client is ready to use immediately and performs no network I/O during
/// construction. The first request through it will trigger the connector,
/// which:
///
/// 1. Completes the TCP+TLS handshake.
/// 2. Generates a fresh random nonce and issues an inline HTTP/1.1
///    `GET /.well-known/tinfoil-attestation?nonce=<hex>` over the *same*
///    connection.
/// 3. Falls back to ATC for the VCEK (the fresh document omits it).
/// 4. Verifies the echoed nonce matches the one sent, the document's
///    `tls_key_fp` matches `sha256(SPKI(peer_cert))`, the embedded
///    certificate matches the peer cert, the document's ECDSA signature
///    validates against it, and the RFC 9266 `tls-exporter` channel binding
///    (when the enclave provides one) equals this session's exporter.
/// 5. Verifies the AMD VCEK chain, the report signature, the TCB floor, the
///    measurement against `allowed_measurements`, and that the report's
///    `REPORT_DATA` equals
///    `sha256(tls_key_fp || hpke_key || nonce || … || tls_exporter)`.
/// 6. Yields the connection to hyper for the real request.
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
        attestation_observer: config.attestation_observer,
        tls_roots,
        require_channel_binding: config.require_channel_binding,
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
/// IPv6 literals are returned with their surrounding brackets so the result
/// is a valid HTTP `Host` header value per RFC 7230.
pub(crate) fn enclave_host(inference_base_url: &str) -> String {
    let after_scheme = match inference_base_url.find("://") {
        Some(scheme_end) => &inference_base_url[scheme_end + 3..],
        None => inference_base_url,
    };
    let authority_end = after_scheme.find('/').unwrap_or(after_scheme.len());
    let authority = &after_scheme[..authority_end];
    // IPv6 literals are bracketed in URL authorities (`[::1]` or `[::1]:8443`).
    // Keep the brackets — they're required in the HTTP `Host` header — and
    // strip only a port that follows the closing bracket. A bare `rfind(':')`
    // would corrupt the address by slicing inside the literal.
    if let Some(rest) = authority.strip_prefix('[') {
        if let Some(close) = rest.find(']') {
            return authority[..close + 2].to_string();
        }
        // Malformed (no closing bracket) — fall through and return as-is.
        return authority.to_string();
    }
    // Regular host[:port].
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

    #[test]
    fn enclave_host_preserves_ipv6_literal() {
        // Bracketed IPv6 with no port — brackets must survive so the result
        // is a valid HTTP Host header value.
        assert_eq!(enclave_host("https://[::1]/v1"), "[::1]");
        assert_eq!(enclave_host("https://[::1]"), "[::1]");
        // Bracketed IPv6 with explicit port — port stripped, brackets kept.
        assert_eq!(enclave_host("https://[::1]:8443/v1"), "[::1]");
        assert_eq!(
            enclave_host("https://[2001:db8::1]:443/foo"),
            "[2001:db8::1]"
        );
        assert_eq!(enclave_host("https://[fe80::1%25eth0]"), "[fe80::1%25eth0]");
    }
}
