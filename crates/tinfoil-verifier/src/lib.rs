//! Tinfoil attestation verification and TLS certificate pinning.
//!
//! Verifies that a Tinfoil inference enclave is running genuine AMD SEV-SNP
//! hardware with expected code measurements, then pins TLS connections to the
//! enclave's attested certificate.
//!
//! Certificate chain verification and report signature checking are delegated
//! to the [`sev`](https://crates.io/crates/sev) crate (virtee/sev).
//!
//! # Prerequisites
//!
//! A rustls `CryptoProvider` must be installed before calling [`verify_and_pin`].
//! The eidolons server does this in `main.rs` via `rustls_rustcrypto::provider()`.

pub mod bundle;
mod error;
mod pinned_client;
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

/// Verify a Tinfoil enclave's attestation and return a TLS-pinned HTTP client.
///
/// 1. Fetches the attestation bundle from the Tinfoil ATC service
/// 2. Verifies the VCEK certificate chain (AMD ARK → ASK → VCEK)
/// 3. Verifies the SEV-SNP attestation report signature
/// 4. Validates TCB policy (minimum firmware versions)
/// 5. Checks the report measurement against `allowed_measurements`
/// 6. Cross-checks the enclave certificate against `report_data`
/// 7. Builds a `reqwest::Client` pinned to the attested TLS public key
///
/// `atc_url` overrides the default ATC endpoint (`https://atc.tinfoil.sh/attestation`).
pub async fn verify_and_pin(
    allowed_measurements: &[&str],
    atc_url: Option<&str>,
) -> Result<(reqwest::Client, Verification), Error> {
    let display_url = atc_url.unwrap_or("atc.tinfoil.sh");
    tracing::info!("Fetching attestation bundle from {display_url}...");
    let attestation_bundle = bundle::fetch_bundle(atc_url).await?;

    // Decode the VCEK certificate from the bundle
    let vcek_der = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &attestation_bundle.vcek,
    )?;

    // Decode the attestation report (base64 → gzip → raw bytes)
    let report_bytes =
        bundle::decode_report(&attestation_bundle.enclave_attestation_report.body)?;

    // Verify chain + report signature (delegated to sev crate)
    tracing::info!("Verifying VCEK certificate chain and report signature...");
    let report = sevsnp::verify_attestation(&vcek_der, &report_bytes)?;

    // Validate TCB policy
    sevsnp::verify_tcb_policy(&report)?;

    // Check measurement against allowed values
    let measurement_hex = hex::encode(report.measurement);
    let matched = allowed_measurements
        .iter()
        .any(|m| m.eq_ignore_ascii_case(&measurement_hex));
    if !matched {
        return Err(Error::MeasurementMismatch {
            measurement: measurement_hex,
            allowed: allowed_measurements.iter().map(|s| s.to_string()).collect(),
        });
    }
    tracing::info!("Measurement verified: {measurement_hex}");

    // Cross-check enclave certificate against report_data
    let tls_fingerprint = &report.report_data[..32];
    sevsnp::verify_enclave_cert_binding(&attestation_bundle.enclave_cert, tls_fingerprint)?;

    let tls_fingerprint_hex = hex::encode(tls_fingerprint);
    tracing::info!("TLS fingerprint verified: {tls_fingerprint_hex}");

    // Build the pinned client
    let mut expected_fp = [0u8; 32];
    expected_fp.copy_from_slice(tls_fingerprint);
    let client = pinned_client::build_pinned_client(expected_fp)?;

    Ok((
        client,
        Verification {
            measurement: measurement_hex,
            tls_fingerprint: tls_fingerprint_hex,
        },
    ))
}
