//! TLS and attestation over a byte stream someone else opened.
//!
//! A host that can open a TCP connection but should take no part in its
//! security — the mobile app's native shell, which hands its wasm build a raw
//! socket — supplies a [`Dialer`]. Everything above the bytes happens here:
//! the TLS handshake (rustls, with the caller's roots), and for an attested
//! endpoint the same [`AttestationCheck::attest_stream`] the native reqwest
//! connector runs, so there is one implementation of what is checked. The
//! host only ever moves ciphertext.
//!
//! Two connectors, one per trust model:
//!
//! - [`AttestingPipe`]: a TLS handshake ([`AttestingPipe::handshake`]) whose
//!   stream is released only by [`Handshaken::attest`], which runs the check
//!   at the time the caller supplies. Built from the same
//!   [`crate::AttestingClientConfig`] as [`crate::attesting_client`].
//! - [`TlsPipe`]: plain TLS to the caller's roots (WebPKI), for endpoints that
//!   carry no attestation.
//!
//! Nothing here keeps time or spawns: the caller bounds each step with its
//! own timer ([`ATTESTATION_DEADLINE`] for the attestation), and frames HTTP
//! over the stream it gets back. On wasm32 nothing need be `Send`; natively
//! everything is, so the same code runs under tokio's multi-threaded runtime.

use std::sync::Arc;

use rustls::pki_types::ServerName;
use tokio::io::{AsyncRead, AsyncWrite};

pub use crate::attest::ATTESTATION_DEADLINE;
use crate::attest::{self, AttestationCheck};
use crate::{AttestingClientConfig, Error};

#[cfg(not(target_family = "wasm"))]
mod bounds {
    /// `Send` natively, nothing on wasm32.
    pub trait MaybeSend: Send {}
    impl<T: Send + ?Sized> MaybeSend for T {}
    /// A boxed future, `Send` natively.
    pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;
}

#[cfg(target_family = "wasm")]
mod bounds {
    /// `Send` natively, nothing on wasm32.
    pub trait MaybeSend {}
    impl<T: ?Sized> MaybeSend for T {}
    /// A boxed future, `Send` natively.
    pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + 'a>>;
}

pub use bounds::{BoxFuture, MaybeSend};

/// A raw, bidirectional byte stream: what a [`Dialer`] opens.
pub trait ByteStream: AsyncRead + AsyncWrite + Unpin + MaybeSend {}
impl<T: AsyncRead + AsyncWrite + Unpin + MaybeSend + ?Sized> ByteStream for T {}

/// A dialed stream, before TLS.
pub type PipeStream = Box<dyn ByteStream>;

/// A TLS session over a dialed stream.
pub type TlsStream = tokio_rustls::client::TlsStream<PipeStream>;

/// Opens a raw byte stream to `host:port`.
///
/// Implemented by whatever embeds the client: the mobile app's bridge to its
/// shell's socket, or [`TcpDialer`] natively. A dialer moves bytes and nothing
/// else; TLS, and every trust decision, are this crate's.
///
/// The dialer itself is shared (`Send + Sync` on every target, so a client
/// holding one stays shareable); on wasm32 the futures and streams it returns
/// need not be `Send`.
pub trait Dialer: Send + Sync {
    fn dial<'a>(&'a self, host: &'a str, port: u16) -> BoxFuture<'a, std::io::Result<PipeStream>>;
}

/// A [`Dialer`] over tokio's `TcpStream`: for native tests and tools.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct TcpDialer;

#[cfg(not(target_family = "wasm"))]
impl Dialer for TcpDialer {
    fn dial<'a>(&'a self, host: &'a str, port: u16) -> BoxFuture<'a, std::io::Result<PipeStream>> {
        Box::pin(async move {
            let host = host.trim_start_matches('[').trim_end_matches(']');
            let stream = tokio::net::TcpStream::connect((host, port)).await?;
            stream.set_nodelay(true)?;
            Ok(Box::new(stream) as PipeStream)
        })
    }
}

/// An attested endpoint reached over dialed streams.
pub struct AttestingPipe {
    tls: Arc<rustls::ClientConfig>,
    check: Arc<AttestationCheck>,
}

impl AttestingPipe {
    /// The connector `config` describes, with its pins compiled now (an
    /// invalid pin is refused here, as [`crate::attesting_client`] refuses
    /// it). No network I/O.
    pub fn new(mut config: AttestingClientConfig<'_>) -> Result<Self, Error> {
        let check = AttestationCheck::from_config(&mut config)?;
        Ok(Self {
            tls: Arc::new(attest::tls_config(config.tls_roots)),
            check: Arc::new(check),
        })
    }

    /// Dial `host:port` and complete the TLS handshake. The stream comes back
    /// only through [`Handshaken::attest`].
    pub async fn handshake(
        &self,
        dialer: &dyn Dialer,
        host: &str,
        port: u16,
    ) -> Result<Handshaken, Error> {
        let (stream, peer_cert_der) = tls_connect(&self.tls, dialer, host, port).await?;
        Ok(Handshaken {
            stream,
            peer_cert_der,
            check: self.check.clone(),
        })
    }
}

/// A TLS session to an attested endpoint that has not been attested yet.
pub struct Handshaken {
    stream: TlsStream,
    peer_cert_der: Vec<u8>,
    check: Arc<AttestationCheck>,
}

impl Handshaken {
    /// Run the attestation check over this session, with the vendor
    /// collateral's validity windows checked at `now` (Unix seconds), and
    /// hand the stream back for the application request. The caller bounds
    /// this with [`ATTESTATION_DEADLINE`].
    pub async fn attest(mut self, now: u64) -> Result<TlsStream, Error> {
        self.check
            .attest_stream(&mut self.stream, &self.peer_cert_der, now)
            .await?;
        Ok(self.stream)
    }
}

/// Plain TLS over dialed streams, to the caller's roots.
pub struct TlsPipe {
    tls: Arc<rustls::ClientConfig>,
}

impl TlsPipe {
    /// TLS to `roots`, ALPN pinned to HTTP/1.1 (the one protocol spoken over
    /// a pipe).
    pub fn new(roots: rustls::RootCertStore) -> Self {
        Self {
            tls: Arc::new(attest::tls_config(roots)),
        }
    }

    /// Dial `host:port` and complete the TLS handshake.
    pub async fn connect(
        &self,
        dialer: &dyn Dialer,
        host: &str,
        port: u16,
    ) -> Result<TlsStream, Error> {
        tls_connect(&self.tls, dialer, host, port)
            .await
            .map(|(stream, _)| stream)
    }
}

/// Dial and complete a TLS handshake, returning the session and the leaf
/// certificate it landed on.
async fn tls_connect(
    config: &Arc<rustls::ClientConfig>,
    dialer: &dyn Dialer,
    host: &str,
    port: u16,
) -> Result<(TlsStream, Vec<u8>), Error> {
    // An IPv6 literal arrives bracketed, as a URL authority writes it.
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let server_name = ServerName::try_from(bare.to_string())
        .map_err(|e| Error::Tls(format!("invalid server name {host:?}: {e}")))?;
    let tcp = dialer
        .dial(host, port)
        .await
        .map_err(|e| Error::Connect(format!("dial {host}:{port}: {e}")))?;
    let stream = tokio_rustls::TlsConnector::from(config.clone())
        .connect(server_name, tcp)
        .await
        .map_err(|e| Error::Connect(format!("TLS handshake with {host}: {e}")))?;
    let peer_cert_der = stream
        .get_ref()
        .1
        .peer_certificates()
        .and_then(|certs| certs.first())
        .ok_or_else(|| Error::Connect("TLS session exposed no peer certificate".to_string()))?
        .as_ref()
        .to_vec();
    Ok((stream, peer_cert_der))
}
