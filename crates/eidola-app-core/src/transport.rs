//! Where this crate's HTTP clients are built, per target.
//!
//! Natively every client is reqwest over rustls with the OS trust store: the
//! attested client the Eidola backend is reached through
//! (`tinfoil_verifier::attesting_client`), the plain client for local engines
//! and external backends, and the updater's. wasm32 has no transport: every
//! builder refuses with `AppError::NoTransport`, so nothing on that target
//! can send a request.

/// Whether a client follows a redirect, or hands it back as the answer.
pub(crate) enum Redirects {
    Follow,
    Refuse,
}

#[cfg(not(target_family = "wasm"))]
mod imp {
    use super::Redirects;
    use crate::error::AppError;

    /// [`tinfoil_verifier::attesting_client`] for the Eidola backend, over
    /// the OS trust store.
    pub(crate) async fn attesting_client(
        allowed_measurements: &[tinfoil_verifier::AllowedMeasurement],
        base_url: &str,
        hardware_root_der: Option<&[u8]>,
        hardware_intermediate_der: Option<&[u8]>,
        attestation_observer: Option<tinfoil_verifier::AttestationObserver>,
    ) -> Result<reqwest::Client, AppError> {
        tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
            allowed_measurements,
            inference_base_url: base_url,
            trusted_ark_der: hardware_root_der,
            trusted_ask_der: hardware_intermediate_der,
            snp_min_tcb: None,
            snp_observer: None,
            attestation_observer,
            tls_roots: native_root_store(),
        })
        .await
        .map_err(|e| AppError::Attestation {
            message: format!("attestation client build failed: {e}"),
        })
    }

    /// A plain (non-attesting) HTTPS-capable client over the OS trust store,
    /// for local engines and external backends
    /// (`local_models::plain_http_client` and its siblings).
    pub(crate) fn plain_client(
        user_agent: Option<&'static str>,
        redirects: Redirects,
    ) -> Result<reqwest::Client, AppError> {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(native_root_store())
            .with_no_client_auth();
        let mut builder = reqwest::Client::builder().tls_backend_preconfigured(tls_config);
        if let Some(agent) = user_agent {
            builder = builder.user_agent(agent);
        }
        if matches!(redirects, Redirects::Refuse) {
            builder = builder.redirect(reqwest::redirect::Policy::none());
        }
        builder.build().map_err(|e| AppError::LocalModel {
            message: format!("constructing HTTP client: {e}"),
        })
    }

    /// The updater's client (`updater::build_http_client`).
    pub(crate) fn update_client() -> Result<reqwest::Client, AppError> {
        // Idempotent — AppCore::new normally installs this; for standalone
        // updater callers (tests, future binaries) this is the safety net.
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(native_root_store())
            .with_no_client_auth();

        reqwest::Client::builder()
            .tls_backend_preconfigured(tls_config)
            .user_agent(concat!("eidola-app-core/", env!("CARGO_PKG_VERSION")))
            // Bounds the handshake only. A whole-request timeout would be
            // wrong here: the same client fetches a small signed document and
            // a container of tens of megabytes, and the second is allowed to
            // take as long as it keeps making progress — which is what the
            // per-read stall bound measures.
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| AppError::Update {
                message: format!("constructing HTTPS client: {e}"),
            })
    }

    /// A body of one chunk that `chunk` produces when the transport first
    /// asks for it: after the connection is open and attested.
    pub(crate) fn deferred_body<F>(chunk: F) -> reqwest::Body
    where
        F: std::future::Future<Output = std::io::Result<bytes::Bytes>> + Send + 'static,
    {
        reqwest::Body::wrap_stream(futures_util::stream::once(chunk))
    }

    fn native_root_store() -> rustls::RootCertStore {
        let mut store = rustls::RootCertStore::empty();
        let native = rustls_native_certs::load_native_certs();
        for cert in native.certs {
            let _ = store.add(cert);
        }
        store
    }
}

#[cfg(target_family = "wasm")]
mod imp {
    use super::Redirects;
    use crate::error::AppError;

    pub(crate) async fn attesting_client(
        _allowed_measurements: &[tinfoil_verifier::AllowedMeasurement],
        _base_url: &str,
        _hardware_root_der: Option<&[u8]>,
        _hardware_intermediate_der: Option<&[u8]>,
        _attestation_observer: Option<tinfoil_verifier::AttestationObserver>,
    ) -> Result<reqwest::Client, AppError> {
        Err(AppError::NoTransport)
    }

    pub(crate) fn plain_client(
        _user_agent: Option<&'static str>,
        _redirects: Redirects,
    ) -> Result<reqwest::Client, AppError> {
        Err(AppError::NoTransport)
    }

    pub(crate) fn update_client() -> Result<reqwest::Client, AppError> {
        Err(AppError::NoTransport)
    }

    /// The fetch-backed reqwest takes a body's bytes up front, so nothing can
    /// wait to be asked for: the chunk is dropped unpolled, and the body is
    /// empty. No client exists here to send it, and one that did would send
    /// nothing of the request.
    pub(crate) fn deferred_body<F>(_chunk: F) -> reqwest::Body
    where
        F: std::future::Future<Output = std::io::Result<bytes::Bytes>> + 'static,
    {
        reqwest::Body::from(Vec::new())
    }
}

pub(crate) use imp::*;
