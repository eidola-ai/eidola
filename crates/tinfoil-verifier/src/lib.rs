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
    let allowed: Vec<String> = config
        .allowed_measurements
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Determine whether to fetch from ATC or the server's own well-known endpoint.
    // When a custom root CA is configured (dev shim), always fetch directly from
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
        bundle::ResolvedAttestation {
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

    // Resolve VCEK: from attestation doc, or fail
    let vcek_der = resolved
        .vcek_der
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

    // Cross-check enclave certificate against report_data (if available)
    let tls_fingerprint = &report.report_data[..32];
    if let Some(enclave_cert) = &resolved.enclave_cert {
        sevsnp::verify_enclave_cert_binding(enclave_cert, tls_fingerprint)?;
    }

    let tls_fingerprint_hex = hex::encode(tls_fingerprint);
    tracing::info!("TLS fingerprint verified: {tls_fingerprint_hex}");

    // Derive the well-known attestation URL from the inference base URL
    tracing::info!("Per-connection attestation URL: {attestation_url}");

    // Build the attesting client, seeded with the initial verified fingerprint and VCEK
    let mut initial_fp = [0u8; 32];
    initial_fp.copy_from_slice(tls_fingerprint);

    let client = attesting_client::build_attesting_client(
        initial_fp,
        report.chip_id,
        vcek_der,
        config.trusted_ark_der.map(|d| d.to_vec()),
        config.trusted_ask_der.map(|d| d.to_vec()),
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

/// Build a reqwest client for bootstrap attestation fetches.
///
/// When a custom trusted ARK is provided, it's added as a TLS root certificate
/// so the client can connect to servers using certs signed by that root (e.g.
/// the dev shim). Without a custom root, standard WebPKI roots are used.
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
