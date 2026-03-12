//! RA-TLS certificate management for dstack TEE attestation.
//!
//! The server terminates TLS itself using a certificate whose X.509 extensions
//! embed an attestation quote from the dstack guest agent. This lets clients
//! cryptographically verify they are talking to code running inside a TEE.

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use dstack_sdk::dstack_client::{DstackClient, TlsKeyConfig};
use rand_core::{OsRng, RngCore};
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;
use tokio_rustls::TlsAcceptor;
use tracing::{info, warn};

/// How often to regenerate the RA-TLS certificate (attestation quotes expire).
const CERT_ROTATION_INTERVAL: Duration = Duration::from_secs(12 * 3600);

/// Random jitter added to the rotation interval.
const CERT_ROTATION_JITTER_SECS: u64 = 3600;

/// A certificate resolver that serves a hot-swappable [`CertifiedKey`].
///
/// Uses [`ArcSwap`] for lock-free reads on every TLS handshake.
#[derive(Debug)]
pub struct RaTlsCertResolver {
    certified_key: ArcSwap<CertifiedKey>,
}

impl RaTlsCertResolver {
    pub fn new(initial: CertifiedKey) -> Arc<Self> {
        Arc::new(Self {
            certified_key: ArcSwap::new(Arc::new(initial)),
        })
    }

    pub fn update(&self, new_key: CertifiedKey) {
        self.certified_key.store(Arc::new(new_key));
    }
}

impl ResolvesServerCert for RaTlsCertResolver {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.certified_key.load_full())
    }
}

/// Fetch an RA-TLS certificate and private key from the dstack guest agent.
pub async fn fetch_ra_tls_cert(
    dstack: &DstackClient,
    sans: &[String],
) -> anyhow::Result<CertifiedKey> {
    let config = TlsKeyConfig::builder()
        .subject("eidolons-server")
        .alt_names(sans.to_vec())
        .usage_ra_tls(true)
        .usage_server_auth(true)
        .build();

    let response = dstack.get_tls_key(config).await?;

    // Parse the PEM-encoded private key.
    let mut key_cursor = std::io::Cursor::new(response.key.as_bytes());
    let private_key = rustls_pemfile::private_key(&mut key_cursor)?
        .ok_or_else(|| anyhow::anyhow!("no private key found in dstack response"))?;

    // Parse PEM-encoded certificate chain.
    let mut certs = Vec::new();
    for pem_str in &response.certificate_chain {
        let mut cursor = std::io::Cursor::new(pem_str.as_bytes());
        for cert in rustls_pemfile::certs(&mut cursor) {
            certs.push(cert?);
        }
    }
    if certs.is_empty() {
        anyhow::bail!("dstack returned empty certificate chain");
    }

    let signing_key = rustls::crypto::CryptoProvider::get_default()
        .ok_or_else(|| anyhow::anyhow!("no rustls crypto provider installed"))?
        .key_provider
        .load_private_key(private_key)?;

    Ok(CertifiedKey::new(certs, signing_key))
}

/// Build a [`TlsAcceptor`] backed by the given certificate resolver.
pub fn build_tls_acceptor(resolver: Arc<RaTlsCertResolver>) -> TlsAcceptor {
    let tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_cert_resolver(resolver);

    TlsAcceptor::from(Arc::new(tls_config))
}

/// Read `TLS_SANS` env var (comma-separated) or fall back to dev defaults.
pub fn parse_sans() -> Vec<String> {
    std::env::var("TLS_SANS")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| vec!["localhost".to_string(), "server".to_string()])
}

/// Periodically regenerate the RA-TLS certificate (attestation quotes expire).
pub fn spawn_cert_rotation_task(
    resolver: Arc<RaTlsCertResolver>,
    dstack: DstackClient,
    sans: Vec<String>,
) {
    tokio::spawn(async move {
        loop {
            let jitter = OsRng.next_u64() % CERT_ROTATION_JITTER_SECS;
            let sleep_dur = CERT_ROTATION_INTERVAL + Duration::from_secs(jitter);
            tokio::time::sleep(sleep_dur).await;

            match fetch_ra_tls_cert(&dstack, &sans).await {
                Ok(new_cert) => {
                    resolver.update(new_cert);
                    info!("RA-TLS certificate rotated");
                }
                Err(e) => {
                    warn!("RA-TLS certificate rotation failed (will retry): {e}");
                }
            }
        }
    });
}
