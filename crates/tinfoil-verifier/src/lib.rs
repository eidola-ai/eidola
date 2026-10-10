//! Tinfoil attestation verification with per-handshake attesting.
//!
//! Verifies that every new TLS connection to a Tinfoil enclave terminates
//! inside genuine confidential-computing hardware running a pinned code
//! measurement. The caller's pins are platform-tagged
//! ([`AllowedMeasurement`]): AMD SEV-SNP evidence is checked only against
//! SEV-SNP entries and Intel TDX evidence only against TDX entries, and
//! evidence from a platform with no entry is refused before it is
//! authenticated. A pin set built from release records
//! ([`EnclaveMeasurement`]) is SEV-SNP-only.
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
//! The v3 document is self-contained. Its signed CPU evidence binds a fresh
//! nonce and the exact bytes of endorsed crypto/device sections; those
//! sections include the TLS SPKI fingerprint and HPKE public key. AMD VCEK
//! and CRL collateral travel in the same document and are verified offline
//! against pinned AMD roots. No attestation transparency service or vendor
//! collateral endpoint is consulted during verification. A relay can replay
//! an older AMD-signed CRL only while its signed validity interval remains
//! current; it cannot alter the list or extend that interval.
//!
//! The nonce prevents replay, and the endorsed TLS SPKI binds the report to
//! the key used by the peer certificate. It still does **not** bind the live
//! TLS session. An attacker holding an exfiltrated TLS private key could MITM
//! this connection while relaying a fresh nonce-bound report from the genuine
//! enclave. Closing that residual gap requires TLS channel binding, such as
//! committing exporter key material into `REPORT_DATA`.
//!
//! SEV-SNP verification is delegated to the [`sev`](https://crates.io/crates/sev)
//! crate and TDX quote verification to [`dcap-qvl`](https://crates.io/crates/dcap-qvl);
//! TDX collateral is the document's captured Intel PCS responses, verified
//! offline against the pinned Intel SGX root. TDX pins follow the IGVM launch
//! model ([`TdxPin`]).
//!
//! # Prerequisites
//!
//! A rustls `CryptoProvider` must be installed before calling [`attesting_client`].
//! The eidola server does this in `main.rs` via `rustls_rustcrypto::provider()`.
//!
//! # TLS root sourcing
//!
//! `tinfoil-verifier` is intentionally agnostic about where TLS trust roots
//! come from. Callers populate [`AttestingClientConfig::tls_roots`] for the
//! attested inference endpoint. Each consumer picks the source that fits its
//! environment:
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
mod device;
mod error;
pub mod measurement;
pub mod sevsnp;
mod sevsnp_crl;
pub mod tdx;

pub use bundle::Platform;
pub use error::Error;
pub use measurement::{
    AllowedMeasurement, EnclaveMeasurement, MatchedMeasurement, PckFlag, PlatformMeasurement,
    TdxLaunchMeasurement, TdxMeasurement, TdxPin, TdxPolicy,
};
pub use sevsnp::{SevSnpObserver, SevSnpTcbObservation, SevSnpTcbPolicy, SevSnpTcbSvns};

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
    /// For TDX: `{mrtd}:{mrconfigid}` (two 48-byte values, colon-separated).
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
    /// Allowed measurements, each tagged with its platform. Evidence is
    /// checked only against entries of its own platform; a platform with no
    /// entry is refused. Convert release records with
    /// `AllowedMeasurement::from(&record)`, which pins SEV-SNP only.
    pub allowed_measurements: &'a [AllowedMeasurement],
    /// Base URL of the inference endpoint (e.g. `https://inference.tinfoil.sh/v1`).
    /// The `/.well-known/tinfoil-attestation` endpoint is derived from the origin.
    pub inference_base_url: &'a str,
    /// TLS root store used for the attested inference endpoint. The verifier
    /// is intentionally agnostic about where these roots come from — the caller decides
    /// whether to populate the store from `webpki-roots`, the OS keychain
    /// via `rustls-native-certs`, a custom PEM, or some union. The server
    /// (running inside an enclave with no system trust store) typically
    /// uses `webpki-roots`; the CLI and macOS app use `rustls-native-certs`
    /// so developers can install local dev CAs in their keychain. Custom
    /// SEV-SNP attestation roots (`trusted_ark_der` / `trusted_ask_der`)
    /// are deliberately *not* added here; they only feed the SEV-SNP chain
    /// verifier.
    pub tls_roots: rustls::RootCertStore,
    /// Optional custom trusted ARK (Root CA) DER bytes. Overrides the
    /// built-in AMD Genoa ARK in the SEV-SNP attestation chain verifier.
    /// **Not** added to any TLS root store; if you need TLS to trust a
    /// custom CA (e.g. for the tinfoil shim mock), install the cert in your
    /// system trust store.
    pub trusted_ark_der: Option<&'a [u8]>,
    /// Optional custom trusted ASK DER bytes. Same caveats as
    /// [`Self::trusted_ark_der`].
    pub trusted_ask_der: Option<&'a [u8]>,
    /// Operator-supplied minimum TCB SVNs the SEV-SNP `reported_tcb`
    /// must satisfy. When `None`, defaults to
    /// [`SevSnpTcbPolicy::amd_recommended`] (`bootloader >= 0x07`,
    /// `snp >= 0x0E`, `microcode >= 0x48`, no `tee` floor). The rollback
    /// check (`reported_tcb >= committed_tcb`) is structural and always
    /// applied regardless of this setting.
    pub snp_min_tcb: Option<SevSnpTcbPolicy>,
    /// Optional observer fired for every SEV-SNP attestation that
    /// completes signature verification, **including ones the policy
    /// rejects**. The callback runs synchronously on the TLS handshake hot
    /// path and must be cheap and non-blocking. Unused on TDX backends.
    pub snp_observer: Option<SevSnpObserver>,
    /// Optional observer fired on every successful attestation verification
    /// for a new TLS connection. Receives the full attestation details
    /// including the raw report bytes, matched measurement, and TLS
    /// binding hash. It has the same hot-path constraints as
    /// [`Self::snp_observer`].
    pub attestation_observer: Option<AttestationObserver>,
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
/// 3. Strictly parses the self-contained v3 envelope and recomputes its
///    endorsed-section hashes and domain-separated `REPORT_DATA`.
/// 4. Verifies the echoed nonce and that the endorsed TLS SPKI fingerprint
///    matches `sha256(SPKI(peer_cert))`.
/// 5. Refuses evidence from a platform with no pinned entry.
/// 6. SEV-SNP: enforces report field hygiene (exact length, version ≥ 3,
///    DEBUG/MIGRATE_MA policy bits off, VCEK-signed, no ID-block), then
///    verifies the document-carried AMD VCEK chain and report, plus the
///    complete carried CRL's ARK signature, identity, half-open validity
///    interval, and ASK/VCEK revocation state; then enforces TCB floor and
///    measurement. TDX: verifies the quote to the pinned Intel root with the
///    document's captured Intel PCS collateral, then the pinned MRTD,
///    MRCONFIGID, zero RTMRs, and machine policy.
/// 7. Requires the signed `REPORT_DATA` to equal the envelope's
///    recomputation, and the matched entry's device-evidence requirement.
/// 8. Yields the connection to hyper for the real request.
///
/// Callers that want fail-fast-at-startup semantics should make one trivial
/// request (e.g. `client.get(format!("{base}/v1/models")).send().await`)
/// after construction and treat its outcome as the readiness check.
pub async fn attesting_client(config: AttestingClientConfig<'_>) -> Result<reqwest::Client, Error> {
    let tls_roots = std::sync::Arc::new(config.tls_roots);

    let snp_policy = config.snp_min_tcb.unwrap_or_default();
    let allowed_measurements = measurement::compile_pins(config.allowed_measurements)?;

    attesting_client::build_attesting_client(attesting_client::BuildParams {
        inference_base_url: config.inference_base_url.to_string(),
        trusted_ark_der: config.trusted_ark_der.map(|d| d.to_vec()),
        trusted_ask_der: config.trusted_ask_der.map(|d| d.to_vec()),
        allowed_measurements,
        snp_policy,
        snp_observer: config.snp_observer,
        attestation_observer: config.attestation_observer,
        tls_roots,
    })
}

/// Validate pins exactly as [`attesting_client`] compiles them, without
/// building a client: every hex field decoded at its width, a TDX policy's
/// safety bits and non-empty MR_SEAM allowlist enforced. A pin that passes
/// here is one `attesting_client` accepts, so a consumer can refuse a bad pin
/// set where it is authored (the gateway does so at build time) rather than
/// at its first connection.
pub fn validate_pins(pins: &[AllowedMeasurement]) -> Result<(), Error> {
    measurement::compile_pins(pins).map(|_| ())
}

/// Extract the bare host (no scheme, no port, no path) from an inference base URL.
///
/// Used as the `Host` header in the per-handshake inline attestation request.
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
