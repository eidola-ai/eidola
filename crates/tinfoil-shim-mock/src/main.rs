//! Tinfoil shim mock — the development binary. Everything it serves is the
//! library's ([`tinfoil_shim_mock::start`]); this reads one shim's
//! configuration from the environment.
//!
//! Environment variables:
//!   UPSTREAM_URL    - upstream server URL (default: http://127.0.0.1:8080)
//!   LISTEN_ADDR     - address to bind (default: 0.0.0.0:8443)
//!   DEV_MEASUREMENT - measurement to advertise (default: 48 zero bytes hex)
//!   CERT_DIR        - directory to store persistent certs (default: .dev-certs)

use std::net::SocketAddr;

use tinfoil_shim_mock::{BoxError, ShimConfig};

/// 48 zero bytes = 96 hex chars (matches SEV-SNP measurement size).
const DEFAULT_MEASUREMENT: &str = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tinfoil_shim_mock=info".parse().unwrap()),
        )
        .init();

    let upstream_url =
        std::env::var("UPSTREAM_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let listen_addr: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8443".to_string())
        .parse()
        .expect("invalid LISTEN_ADDR");
    let measurement =
        std::env::var("DEV_MEASUREMENT").unwrap_or_else(|_| DEFAULT_MEASUREMENT.to_string());
    let cert_dir = std::env::var("CERT_DIR").unwrap_or_else(|_| ".dev-certs".to_string());
    let measurement = hex::decode(&measurement).expect("invalid DEV_MEASUREMENT hex");

    let shim = tinfoil_shim_mock::start(ShimConfig {
        upstream_url,
        listen_addr,
        measurement,
        cert_dir: cert_dir.into(),
        gpus: 0,
    })
    .await?;
    shim.wait().await;
    Err("the listener stopped".into())
}
