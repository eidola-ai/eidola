//! Account billing operations shared by the gateway and private support tooling.
//!
//! No account credentials, ACT proofs, nullifiers or inference data belong here.
//! Database transactions own credit effects; Stripe observations reconcile them.

pub mod connection;
pub mod error;
pub mod service;
pub mod store;
pub mod stripe;
pub use error::BillingError;

pub fn tls_config() -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth()
}
