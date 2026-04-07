//! Tinfoil attestation verification with per-connection attesting.
//!
//! Verifies that Tinfoil inference enclaves are running genuine AMD SEV-SNP or
//! Intel TDX hardware with expected code measurements, then verifies attestation
//! on each new TLS connection (caching verified fingerprints for reconnections).
//!
//! SEV-SNP verification is delegated to the [`sev`](https://crates.io/crates/sev)
//! crate. TDX Quote V4 verification is delegated to [`dcap_qvl`].
//!
//! # Prerequisites
//!
//! A rustls `CryptoProvider` must be installed before calling [`attesting_client`].
//! The eidola server does this in `main.rs` via `rustls_rustcrypto::provider()`.

mod attesting_client;
pub mod bundle;
mod error;
pub mod measurement;
pub mod sevsnp;
pub mod tdx;

pub use bundle::Platform;
pub use error::Error;
pub use measurement::{EnclaveMeasurement, MatchedMeasurement, TdxMeasurement};

/// Result of a successful attestation verification.
#[derive(Debug, Clone)]
pub struct Verification {
    /// The measurement that was observed and matched. Carries the platform
    /// implicitly via the [`MatchedMeasurement`] variant.
    pub measurement: MatchedMeasurement,
    /// Hex-encoded SHA-256 fingerprint of the enclave's TLS public key.
    pub tls_fingerprint: String,
}

/// Configuration for [`attesting_client`].
pub struct AttestingClientConfig<'a> {
    /// Allowed enclave releases. Each entry pairs a SEV-SNP measurement with a
    /// TDX measurement; the verifier picks the matching field based on the
    /// platform observed in the attestation document.
    pub allowed_measurements: &'a [EnclaveMeasurement],
    /// Base URL of the inference endpoint (e.g. `https://inference.tinfoil.sh/v1`).
    /// The `/.well-known/tinfoil-attestation` endpoint is derived from the origin.
    pub inference_base_url: &'a str,
    /// Optional ATC URL override for initial bootstrap verification.
    pub atc_url: Option<&'a str>,
    /// Optional custom trusted ARK (Root CA) DER bytes.
    /// When set, this overrides the built-in AMD Genoa ARK for chain verification
    /// and is also added as a TLS root certificate for bootstrap connections.
    pub trusted_ark_der: Option<&'a [u8]>,
    /// Optional custom trusted ASK DER bytes.
    pub trusted_ask_der: Option<&'a [u8]>,
}

/// Build an attesting HTTP client that verifies enclave attestation per-connection.
///
/// 1. Fetches the attestation bundle from ATC for initial bootstrap verification
/// 2. Verifies VCEK chain, report signature, TCB policy, and measurement
/// 3. Returns a `reqwest::Client` that verifies attestation on each new TLS
///    connection, caching verified fingerprints for fast reconnections
///
/// Unlike static cert pinning, this approach handles load-balanced deployments:
/// each new connection fetches `/.well-known/tinfoil-attestation` from the
/// connected instance to verify its attestation before proceeding.
pub async fn attesting_client(
    config: AttestingClientConfig<'_>,
) -> Result<(reqwest::Client, Verification), Error> {
    let attestation_url = well_known_url(config.inference_base_url);
    let allowed: Vec<EnclaveMeasurement> = config.allowed_measurements.to_vec();

    // Determine whether to fetch from ATC or the server's own well-known endpoint.
    // When a custom root CA is configured (tinfoil shim mock), always fetch directly from
    // the server — ATC only knows about the production Tinfoil enclave.
    // Also use direct fetch when atc_url explicitly matches the well-known URL.
    let use_direct = config.trusted_ark_der.is_some()
        || config
            .atc_url
            .map(|url| url == attestation_url)
            .unwrap_or(false);

    // Fetch attestation: direct (v3 → v2 fallback) or ATC bundle
    let resolved = if use_direct {
        let client = build_bootstrap_client(config.trusted_ark_der)?;
        tracing::info!("Fetching attestation from {attestation_url} (direct)...");
        bundle::fetch_well_known(&client, &attestation_url).await?
    } else {
        let display_url = config.atc_url.unwrap_or(bundle::DEFAULT_ATC_URL);
        tracing::info!("Fetching attestation bundle from {display_url} for bootstrap...");
        let atc_bundle = bundle::fetch_bundle(config.atc_url).await?;
        let platform = bundle::platform_from_format(&atc_bundle.enclave_attestation_report.format)?;
        bundle::ResolvedAttestation {
            platform,
            report_bytes: bundle::decode_report_gzipped(
                &atc_bundle.enclave_attestation_report.body,
            )?,
            vcek_der: Some(sevsnp::decode_base64(&atc_bundle.vcek)?),
            ark_der: atc_bundle
                .ark
                .as_ref()
                .and_then(|s| sevsnp::decode_base64(s).ok()),
            ask_der: atc_bundle
                .ask
                .as_ref()
                .and_then(|s| sevsnp::decode_base64(s).ok()),
            enclave_cert: Some(atc_bundle.enclave_cert),
        }
    };

    tracing::info!("Platform: {:?}", resolved.platform);

    // Dispatch verification based on platform
    let (matched, tls_fingerprint, chip_id, vcek_der) = match resolved.platform {
        bundle::Platform::SevSnp => verify_snp_bootstrap(&config, &resolved)?,
        bundle::Platform::Tdx => {
            let bootstrap_client = build_bootstrap_client(config.trusted_ark_der)?;
            verify_tdx_bootstrap(&config, &resolved, &bootstrap_client).await?
        }
    };

    // Cross-check enclave certificate against report_data (if available).
    // This is a defense-in-depth sanity check — the primary binding is enforced
    // per-connection in the TLS verifier. The check is platform-independent:
    // SHA256(SPKI(enclave_cert)) must equal report_data[0..32].
    if let Some(enclave_cert) = &resolved.enclave_cert {
        sevsnp::verify_enclave_cert_binding(enclave_cert, &tls_fingerprint)?;
    }

    let tls_fingerprint_hex = hex::encode(tls_fingerprint);
    tracing::info!("TLS fingerprint verified: {tls_fingerprint_hex}");

    // Derive the well-known attestation URL from the inference base URL
    tracing::info!("Per-connection attestation URL: {attestation_url}");

    // Build the attesting client, seeded with the initial verified fingerprint
    let client = attesting_client::build_attesting_client(
        tls_fingerprint,
        chip_id,
        vcek_der,
        config.trusted_ark_der.map(|d| d.to_vec()),
        config.trusted_ask_der.map(|d| d.to_vec()),
        allowed,
        attestation_url,
    )?;

    Ok((
        client,
        Verification {
            measurement: matched,
            tls_fingerprint: tls_fingerprint_hex,
        },
    ))
}

/// Bootstrap verification result: (matched_measurement, tls_fingerprint, chip_id, vcek_der).
type BootstrapResult = (MatchedMeasurement, [u8; 32], [u8; 64], Vec<u8>);

/// SEV-SNP bootstrap verification.
fn verify_snp_bootstrap(
    config: &AttestingClientConfig<'_>,
    resolved: &bundle::ResolvedAttestation,
) -> Result<BootstrapResult, Error> {
    // Resolve VCEK: from attestation doc, or fail
    let vcek_der = resolved
        .vcek_der
        .clone()
        .ok_or_else(|| Error::Bundle("no VCEK in attestation (v3 endpoint may not include it yet — set attestation_url to ATC as fallback)".to_string()))?;

    // Resolve ARK/ASK: prefer trusted config, fall back to doc, then built-in Genoa
    let ark_der = config.trusted_ark_der.or(resolved.ark_der.as_deref());
    let ask_der = config.trusted_ask_der.or(resolved.ask_der.as_deref());

    // Verify chain + report signature (delegated to sev crate)
    tracing::info!("Verifying VCEK certificate chain and report signature...");
    let report = sevsnp::verify_attestation(&vcek_der, &resolved.report_bytes, ark_der, ask_der)?;

    // Validate TCB policy
    sevsnp::verify_tcb_policy(&report)?;

    // Check measurement against allowed values
    let measurement_hex = hex::encode(report.measurement);
    let matched = check_snp_measurement(config.allowed_measurements, &measurement_hex)?;
    tracing::info!("Measurement verified: {matched}");

    let mut tls_fingerprint = [0u8; 32];
    tls_fingerprint.copy_from_slice(&report.report_data[..32]);

    Ok((matched, tls_fingerprint, report.chip_id, vcek_der))
}

/// TDX bootstrap verification.
///
/// `chip_id` and `vcek_der` are returned as empty/dummy values since TDX doesn't use
/// AMD's VCEK/chip_id system. The attesting client handles this by detecting the platform.
async fn verify_tdx_bootstrap(
    config: &AttestingClientConfig<'_>,
    resolved: &bundle::ResolvedAttestation,
    http_client: &reqwest::Client,
) -> Result<BootstrapResult, Error> {
    tracing::info!("Fetching TDX collateral from Intel PCS...");
    let collateral = tdx::fetch_collateral(http_client, &resolved.report_bytes).await?;

    tracing::info!("Verifying TDX quote...");
    let tdx_result = tdx::verify_quote(&resolved.report_bytes, &collateral)?;

    // Check RTMR1/RTMR2 against allowed values
    let rtmr1_hex = hex::encode(tdx_result.rtmr1);
    let rtmr2_hex = hex::encode(tdx_result.rtmr2);
    let matched = check_tdx_measurement(config.allowed_measurements, &rtmr1_hex, &rtmr2_hex)?;
    tracing::info!("Measurement verified: {matched}");

    let mut tls_fingerprint = [0u8; 32];
    tls_fingerprint.copy_from_slice(&tdx_result.report_data[..32]);

    // TDX doesn't use VCEK/chip_id — return empty placeholders.
    // The per-connection verifier detects TDX from the attestation doc platform field.
    Ok((matched, tls_fingerprint, [0u8; 64], Vec::new()))
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

/// Build a reqwest client for bootstrap attestation fetches.
///
/// When a custom trusted ARK is provided, it's added as a TLS root certificate
/// so the client can connect to servers using certs signed by that root (e.g.
/// the tinfoil shim mock). Without a custom root, standard WebPKI roots are used.
fn build_bootstrap_client(trusted_ark_der: Option<&[u8]>) -> Result<reqwest::Client, Error> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    if let Some(ark_der) = trusted_ark_der {
        root_store
            .add(ark_der.into())
            .map_err(|e| Error::Tls(format!("invalid ARK cert for TLS root: {e}")))?;
    }

    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(30))
        .tls_backend_preconfigured(tls_config)
        .build()
        .map_err(|e| Error::Tls(format!("failed to build bootstrap client: {e}")))
}

/// Derive `/.well-known/tinfoil-attestation` URL from an inference base URL.
///
/// Strips any path (e.g. `/v1`) and appends the well-known path to the origin.
fn well_known_url(inference_base_url: &str) -> String {
    // Parse to extract just the origin (scheme + authority)
    match inference_base_url.find("://") {
        Some(scheme_end) => {
            let after_scheme = &inference_base_url[scheme_end + 3..];
            let authority_end = after_scheme.find('/').unwrap_or(after_scheme.len());
            let origin = &inference_base_url[..scheme_end + 3 + authority_end];
            format!("{origin}/.well-known/tinfoil-attestation")
        }
        None => format!("{inference_base_url}/.well-known/tinfoil-attestation"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_known_url_strips_path() {
        assert_eq!(
            well_known_url("https://inference.tinfoil.sh/v1"),
            "https://inference.tinfoil.sh/.well-known/tinfoil-attestation"
        );
    }

    #[test]
    fn well_known_url_no_path() {
        assert_eq!(
            well_known_url("https://inference.tinfoil.sh"),
            "https://inference.tinfoil.sh/.well-known/tinfoil-attestation"
        );
    }

    #[test]
    fn well_known_url_trailing_slash() {
        assert_eq!(
            well_known_url("https://inference.tinfoil.sh/"),
            "https://inference.tinfoil.sh/.well-known/tinfoil-attestation"
        );
    }
}
