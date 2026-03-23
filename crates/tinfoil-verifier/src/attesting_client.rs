//! Connection-level attestation verification.
//!
//! Instead of pinning to a single pre-fetched certificate fingerprint, this
//! verifier attests each new TLS connection on demand. Verified fingerprints
//! are cached so that reconnections to already-attested instances are fast.
//!
//! VCEKs (per-chip endorsement keys) are cached by chip ID. When a report is
//! signed by an unknown chip, the VCEK is fetched from AMD's Key Distribution
//! Service and verified against the hardcoded Genoa ARK/ASK root certs before
//! being cached.

use std::sync::Arc;
use std::time::Duration;

use dashmap::{DashMap, DashSet};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{
    CryptoProvider, WebPkiSupportedAlgorithms, verify_tls12_signature, verify_tls13_signature,
};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};

use crate::{bundle, sevsnp};

/// A rustls `ServerCertVerifier` that verifies attestation for each new TLS
/// connection. Fingerprints of previously-attested instances are cached so
/// subsequent handshakes to the same instance are instant.
#[derive(Debug)]
struct AttestingVerifier {
    /// SPKI SHA-256 fingerprints that have been attested and verified.
    verified: DashSet<[u8; 32]>,
    /// Cached VCEK DER bytes keyed by chip_id.
    vcek_cache: DashMap<[u8; 64], Vec<u8>>,
    /// Optional custom trusted ARK (Root CA) DER bytes.
    trusted_ark_der: Option<Vec<u8>>,
    /// Optional custom trusted ASK DER bytes.
    trusted_ask_der: Option<Vec<u8>>,
    /// Allowed code measurements (hex-encoded).
    allowed_measurements: Vec<String>,
    /// URL to fetch attestation from (`/.well-known/tinfoil-attestation`).
    attestation_url: String,
    /// Supported TLS signature algorithms.
    supported_algs: WebPkiSupportedAlgorithms,
}

/// Maximum number of attestation fetch attempts when verifying a new fingerprint.
/// Each attempt may hit a different load-balanced instance, so we retry to
/// converge on the target instance's attestation.
const MAX_VERIFY_ATTEMPTS: usize = 3;

impl AttestingVerifier {
    /// Resolve or fetch the VCEK for a given report's chip.
    ///
    /// Returns the cached VCEK if available, otherwise fetches from AMD KDS,
    /// verifies the chain against hardcoded root certs, and caches it.
    async fn resolve_vcek(
        &self,
        report: &sev::firmware::guest::AttestationReport,
        fetch_client: &reqwest::Client,
    ) -> Result<Vec<u8>, crate::Error> {
        // Fast path: already cached for this chip
        if let Some(vcek) = self.vcek_cache.get(&report.chip_id) {
            return Ok(vcek.clone());
        }

        // Fetch from AMD KDS
        let url = sevsnp::kds_vcek_url(report);
        tracing::info!(
            chip_id = hex::encode(report.chip_id),
            "fetching VCEK from AMD KDS"
        );

        let vcek_der = fetch_client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec();

        // Verify the fetched VCEK chains to trusted/hardcoded ARK/ASK before trusting it.
        // verify_report builds the full chain and checks ARK → ASK → VCEK signatures.
        // We pass the report to also verify the report signature, but the chain
        // verification is what validates the VCEK itself.
        sevsnp::verify_report(
            &vcek_der,
            report,
            self.trusted_ark_der.as_deref(),
            self.trusted_ask_der.as_deref(),
        )?;

        // Cache for future use
        self.vcek_cache.insert(report.chip_id, vcek_der.clone());
        tracing::info!(
            chip_id = hex::encode(report.chip_id),
            "cached VCEK from AMD KDS"
        );

        Ok(vcek_der)
    }

    /// Fetch attestation from the well-known endpoint and verify it matches
    /// `target_fp`. Adds any successfully-verified fingerprint to the cache
    /// (even if it's not the target — it's still a valid attested instance).
    async fn verify_new_fingerprint(&self, target_fp: [u8; 32]) -> Result<(), crate::Error> {
        // Ephemeral client for fetching attestation documents. When a custom
        // trusted root is configured (dev shim), it's added so TLS works with
        // the dev shim's self-signed cert chain.
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30));
        if let Some(ark_der) = &self.trusted_ark_der
            && let Ok(cert) = reqwest::Certificate::from_der(ark_der)
        {
            builder = builder.add_root_certificate(cert);
        }
        let fetch_client = builder
            .build()
            .map_err(|e| crate::Error::Tls(format!("ephemeral client: {e}")))?;

        for attempt in 0..MAX_VERIFY_ATTEMPTS {
            // Fetch attestation (tries v3 with embedded VCEK, falls back to v2)
            let resolved = bundle::fetch_well_known(&fetch_client, &self.attestation_url).await?;

            let report_bytes = &resolved.report_bytes;
            let report = sevsnp::parse_report(report_bytes)?;

            // Resolve VCEK: prefer embedded (v3), fall back to cache/AMD KDS
            let vcek_der = match resolved.vcek_der {
                Some(vcek) => vcek,
                None => self.resolve_vcek(&report, &fetch_client).await?,
            };

            // Resolve ARK/ASK: prefer trusted config, fall back to doc
            let ark_der = self
                .trusted_ark_der
                .as_deref()
                .or(resolved.ark_der.as_deref());
            let ask_der = self
                .trusted_ask_der
                .as_deref()
                .or(resolved.ask_der.as_deref());

            // Verify report signature against the VCEK
            sevsnp::verify_report(&vcek_der, &report, ark_der, ask_der)?;
            sevsnp::verify_tcb_policy(&report)?;

            // Check measurement
            let measurement_hex = hex::encode(report.measurement);
            let matched = self
                .allowed_measurements
                .iter()
                .any(|m| m.eq_ignore_ascii_case(&measurement_hex));
            if !matched {
                return Err(crate::Error::MeasurementMismatch {
                    measurement: measurement_hex,
                    allowed: self.allowed_measurements.clone(),
                });
            }

            // Extract the attested fingerprint from report_data[0..32]
            let attested_fp: [u8; 32] = report.report_data[..32]
                .try_into()
                .expect("report_data is 64 bytes");

            // Cache this verified fingerprint
            self.verified.insert(attested_fp);
            tracing::info!(
                fingerprint = hex::encode(attested_fp),
                measurement = measurement_hex,
                attempt,
                "verified enclave instance"
            );

            if attested_fp == target_fp {
                return Ok(());
            }

            tracing::debug!(
                target = hex::encode(target_fp),
                got = hex::encode(attested_fp),
                "attestation fetch hit different instance, retrying"
            );
        }

        // Another thread may have verified our target concurrently
        if self.verified.contains(&target_fp) {
            return Ok(());
        }

        Err(crate::Error::Tls(format!(
            "could not verify fingerprint {} after {MAX_VERIFY_ATTEMPTS} attempts",
            hex::encode(target_fp)
        )))
    }
}

impl ServerCertVerifier for AttestingVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        let fingerprint = sevsnp::sha256_spki_from_der(end_entity.as_ref())
            .map_err(|e| TlsError::General(format!("SPKI fingerprint: {e}")))?;

        // Fast path: already verified
        if self.verified.contains(&fingerprint) {
            return Ok(ServerCertVerified::assertion());
        }

        // Slow path: fetch and verify attestation (blocks the handshake)
        let result = tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(self.verify_new_fingerprint(fingerprint))
        });

        match result {
            Ok(()) => Ok(ServerCertVerified::assertion()),
            Err(e) => Err(TlsError::General(format!(
                "attestation verification failed: {e}"
            ))),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls12_signature(message, cert, dss, &self.supported_algs)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls13_signature(message, cert, dss, &self.supported_algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.supported_algs.supported_schemes()
    }
}

/// Build a `reqwest::Client` that verifies attestation on each new TLS connection.
///
/// `initial_fingerprint` is seeded into the verified cache so the first connection
/// to the initially-attested instance doesn't trigger a redundant fetch.
/// `initial_vcek_der` is seeded into the VCEK cache for the given `initial_chip_id`.
pub fn build_attesting_client(
    initial_fingerprint: [u8; 32],
    initial_chip_id: [u8; 64],
    initial_vcek_der: Vec<u8>,
    trusted_ark_der: Option<Vec<u8>>,
    trusted_ask_der: Option<Vec<u8>>,
    allowed_measurements: Vec<String>,
    attestation_url: String,
) -> Result<reqwest::Client, crate::Error> {
    let provider = CryptoProvider::get_default()
        .ok_or_else(|| crate::Error::Tls("no rustls CryptoProvider installed".into()))?;

    let verified = DashSet::new();
    verified.insert(initial_fingerprint);

    let vcek_cache = DashMap::new();
    vcek_cache.insert(initial_chip_id, initial_vcek_der);

    let verifier = AttestingVerifier {
        verified,
        vcek_cache,
        trusted_ark_der,
        trusted_ask_der,
        allowed_measurements,
        attestation_url,
        supported_algs: provider.signature_verification_algorithms,
    };

    let tls_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_no_client_auth();

    reqwest::Client::builder()
        .use_preconfigured_tls(tls_config)
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(usize::MAX)
        .build()
        .map_err(|e| crate::Error::Tls(format!("failed to build attesting client: {e}")))
}
