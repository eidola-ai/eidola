//! Tinfoil attestation verification with per-connection attesting.
//!
//! Verifies that Tinfoil inference enclaves are running genuine AMD SEV-SNP
//! hardware with expected code measurements, then verifies attestation on
//! each new TLS connection (caching verified fingerprints for reconnections).
//!
//! Certificate chain verification and report signature checking are delegated
//! to the [`sev`](https://crates.io/crates/sev) crate (virtee/sev).
//!
//! # Prerequisites
//!
//! A rustls `CryptoProvider` must be installed before calling [`attesting_client`].
//! The eidolons server does this in `main.rs` via `rustls_rustcrypto::provider()`.

mod attesting_client;
pub mod bundle;
mod error;
pub mod sevsnp;

pub use error::Error;

/// Result of a successful attestation verification.
#[derive(Debug, Clone)]
pub struct Verification {
    /// Hex-encoded measurement that matched an allowed value.
    pub measurement: String,
    /// Hex-encoded SHA-256 fingerprint of the enclave's TLS public key.
    pub tls_fingerprint: String,
}

/// Configuration for [`attesting_client`].
pub struct AttestingClientConfig<'a> {
    /// Hex-encoded allowed measurements.
    pub allowed_measurements: &'a [&'a str],
    /// Base URL of the inference endpoint (e.g. `https://inference.tinfoil.sh/v1`).
    /// The `/.well-known/tinfoil-attestation` endpoint is derived from the origin.
    pub inference_base_url: &'a str,
    /// Optional ATC URL override for initial bootstrap verification.
    pub atc_url: Option<&'a str>,
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
    let display_url = config.atc_url.unwrap_or("atc.tinfoil.sh");
    tracing::info!("Fetching attestation bundle from {display_url} for bootstrap...");
    let attestation_bundle = bundle::fetch_bundle(config.atc_url).await?;

    // Decode the VCEK certificate from the bundle
    let vcek_der = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &attestation_bundle.vcek,
    )?;

    // Decode the attestation report (base64 → gzip → raw bytes)
    let report_bytes = bundle::decode_report(&attestation_bundle.enclave_attestation_report.body)?;

    // Verify chain + report signature (delegated to sev crate)
    tracing::info!("Verifying VCEK certificate chain and report signature...");
    let report = sevsnp::verify_attestation(&vcek_der, &report_bytes)?;

    // Validate TCB policy
    sevsnp::verify_tcb_policy(&report)?;

    // Check measurement against allowed values
    let measurement_hex = hex::encode(report.measurement);
    let matched = config
        .allowed_measurements
        .iter()
        .any(|m| m.eq_ignore_ascii_case(&measurement_hex));
    if !matched {
        return Err(Error::MeasurementMismatch {
            measurement: measurement_hex,
            allowed: config
                .allowed_measurements
                .iter()
                .map(|s| s.to_string())
                .collect(),
        });
    }
    tracing::info!("Measurement verified: {measurement_hex}");

    // Cross-check enclave certificate against report_data
    let tls_fingerprint = &report.report_data[..32];
    sevsnp::verify_enclave_cert_binding(&attestation_bundle.enclave_cert, tls_fingerprint)?;

    let tls_fingerprint_hex = hex::encode(tls_fingerprint);
    tracing::info!("TLS fingerprint verified: {tls_fingerprint_hex}");

    // Derive the well-known attestation URL from the inference base URL
    let attestation_url = well_known_url(config.inference_base_url);
    tracing::info!("Per-connection attestation URL: {attestation_url}");

    // Build the attesting client, seeded with the initial verified fingerprint and VCEK
    let mut initial_fp = [0u8; 32];
    initial_fp.copy_from_slice(tls_fingerprint);

    let allowed = config
        .allowed_measurements
        .iter()
        .map(|s| s.to_string())
        .collect();

    let client = attesting_client::build_attesting_client(
        initial_fp,
        report.chip_id,
        vcek_der,
        allowed,
        attestation_url,
    )?;

    Ok((
        client,
        Verification {
            measurement: measurement_hex,
            tls_fingerprint: tls_fingerprint_hex,
        },
    ))
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
