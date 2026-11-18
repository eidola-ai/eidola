//! Database connections use the caller's PostgreSQL identity and verified TLS.
//!
//! Every connection is TLS, with the server's certificate chain and hostname
//! verified: there is no plaintext mode, for production, development or
//! tests. A database whose certificate does not chain to a WebPKI root is
//! trusted through `DATABASE_SSL_CERT`, which is how a local development
//! database and the test database are reached (`scripts/dev-postgres-tls.sh`).

use std::sync::Arc;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use rustls_pki_types::CertificateDer;
use rustls_pki_types::pem::PemObject;
use tokio_postgres::config::SslMode;
use tokio_postgres_rustls::MakeRustlsConnect;

/// Create a connection pool from a PostgreSQL connection string.
pub fn create_pool(
    database_url: &str,
    database_password: Option<&str>,
    database_ssl_cert: Option<&str>,
) -> Result<Pool, String> {
    let mut pg_config = parse_database_url(database_url)?;
    let tls = tls_connector(database_ssl_cert)?;

    if let Some(database_password) = database_password.filter(|value| !value.is_empty()) {
        pg_config.password(database_password);
    }

    // `Verified` issues a cheap liveness check before handing a connection
    // out of the pool. It costs one round trip per checkout, and keeps a
    // socket the server has dropped (a managed database's resize, failover
    // or idle timeout) from being handed to a request.
    let manager_config = ManagerConfig {
        recycling_method: RecyclingMethod::Verified,
    };
    let manager = Manager::from_config(pg_config, tls, manager_config);

    Pool::builder(manager)
        .max_size(8)
        .build()
        .map_err(|e| format!("failed to build connection pool: {}", e))
}

/// Parse a connection string into a configuration that requires TLS.
///
/// The TLS is rustls with the certificate chain and the server's hostname
/// verified, so `require`, `verify-ca` and `verify-full` all mean libpq's
/// `verify-full` here; the latter two are rewritten to `require` because the
/// driver does not parse them. `prefer`, which is also the driver's default
/// when no `sslmode` is given, is read as `require`: it would fall back to
/// plaintext whenever the other end declines TLS, and on the network path
/// that end can be anyone, who can then ask for the password in cleartext.
/// `disable` is refused.
pub fn parse_database_url(database_url: &str) -> Result<tokio_postgres::Config, String> {
    let normalized_url = database_url
        .replace("sslmode=verify-full", "sslmode=require")
        .replace("sslmode=verify-ca", "sslmode=require");

    let mut pg_config: tokio_postgres::Config = normalized_url
        .parse()
        .map_err(|e| format!("invalid DATABASE_URL: {}", e))?;

    match pg_config.get_ssl_mode() {
        SslMode::Disable => {
            return Err("invalid DATABASE_URL: sslmode=disable is refused; every \
                 database connection is verified TLS (use sslmode=verify-full, and \
                 DATABASE_SSL_CERT for a database outside the WebPKI)"
                .to_string());
        }
        SslMode::Prefer => {
            pg_config.ssl_mode(SslMode::Require);
        }
        _ => {}
    }
    Ok(pg_config)
}

/// The TLS every database connection is made with: the compiled-in WebPKI
/// roots, plus the roots in `database_ssl_cert` (PEM) when given. The extra
/// roots are trusted alongside the WebPKI's, not instead of them.
pub fn tls_connector(database_ssl_cert: Option<&str>) -> Result<MakeRustlsConnect, String> {
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

    let config =
        rustls::ClientConfig::builder_with_provider(Arc::new(rustls_rustcrypto::provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("database TLS configuration: {e}"))?
            .with_root_certificates(root_store)
            .with_no_client_auth();
    Ok(MakeRustlsConnect::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ssl_mode(url: &str) -> SslMode {
        parse_database_url(url).unwrap().get_ssl_mode()
    }

    #[test]
    fn every_mode_but_disable_requires_tls() {
        for url in [
            "postgres://u@db.example.com/d",
            "postgres://u@db.example.com/d?sslmode=prefer",
            "postgres://u@db.example.com/d?sslmode=require",
            "postgres://u@db.example.com/d?sslmode=verify-ca",
            "postgres://u@db.example.com/d?sslmode=verify-full",
            "host=db.example.com user=u dbname=d",
            "host=db.example.com user=u dbname=d sslmode=verify-full",
        ] {
            assert_eq!(ssl_mode(url), SslMode::Require, "{url}");
        }
    }

    #[test]
    fn plaintext_is_refused() {
        for url in [
            "postgres://u@localhost/d?sslmode=disable",
            "host=localhost user=u dbname=d sslmode=disable",
        ] {
            let error = parse_database_url(url).unwrap_err();
            assert!(
                error.contains("sslmode=disable is refused"),
                "{url}: {error}"
            );
            assert!(create_pool(url, None, None).is_err(), "{url}");
        }
    }

    #[test]
    fn options_the_driver_does_not_know_are_refused() {
        // libpq's spelling of "trust the system store" is not an option the
        // driver parses; the compiled-in WebPKI roots are its equivalent.
        assert!(
            parse_database_url(
                "postgres://u@db.example.com/d?sslmode=verify-full&sslrootcert=system"
            )
            .is_err()
        );
        assert!(parse_database_url("postgres://u@db.example.com/d?sslmode=allow").is_err());
    }
}
