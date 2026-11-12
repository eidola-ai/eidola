//! Database connections use the caller's PostgreSQL identity and verified TLS.

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use rustls_pki_types::CertificateDer;
use rustls_pki_types::pem::PemObject;
use tokio_postgres::NoTls;
use tokio_postgres::config::SslMode;
use tokio_postgres_rustls::MakeRustlsConnect;

/// Create a connection pool from a PostgreSQL connection string.
pub fn create_pool(
    database_url: &str,
    database_password: Option<&str>,
    database_ssl_cert: Option<&str>,
) -> Result<Pool, String> {
    let normalized_url = database_url.replace("sslmode=verify-full", "sslmode=require");
    let normalized_url = normalized_url.replace("sslmode=verify-ca", "sslmode=require");

    let mut pg_config: tokio_postgres::Config = normalized_url
        .parse()
        .map_err(|e| format!("invalid DATABASE_URL: {}", e))?;

    if let Some(database_password) = database_password.filter(|value| !value.is_empty()) {
        pg_config.password(database_password);
    }

    // `Verified` issues a cheap liveness check before handing a connection
    // out of the pool. This costs one extra round-trip per checkout but is
    // required when the upstream is a serverless Postgres (e.g. Neon) whose
    // compute can autosuspend and silently kill long-lived sockets.
    let manager_config = ManagerConfig {
        recycling_method: RecyclingMethod::Verified,
    };
    let manager = match pg_config.get_ssl_mode() {
        SslMode::Disable => Manager::from_config(pg_config, NoTls, manager_config),
        _ => {
            let tls = MakeRustlsConnect::new(build_tls_config(database_ssl_cert)?);
            Manager::from_config(pg_config, tls, manager_config)
        }
    };

    Pool::builder(manager)
        .max_size(8)
        .build()
        .map_err(|e| format!("failed to build connection pool: {}", e))
}

fn build_tls_config(database_ssl_cert: Option<&str>) -> Result<rustls::ClientConfig, String> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    if let Some(database_ssl_cert) = database_ssl_cert.filter(|value| !value.is_empty()) {
        let certificates = CertificateDer::pem_slice_iter(database_ssl_cert.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("invalid DATABASE_SSL_CERT PEM: {e:?}"))?;

        if certificates.is_empty() {
            return Err("DATABASE_SSL_CERT did not contain any PEM certificates".to_string());
        }

        let (added, ignored) = root_store.add_parsable_certificates(certificates);
        if added == 0 {
            return Err(
                "DATABASE_SSL_CERT did not contain any usable root certificates".to_string(),
            );
        }
        if ignored > 0 {
            tracing::warn!("Ignored {ignored} invalid certificate(s) in DATABASE_SSL_CERT");
        }
    }

    Ok(rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth())
}
