//! The byte-pipe transport (`tinfoil_verifier::pipe`) against an in-process
//! `tinfoil-shim-mock`, over a `TcpDialer`: TLS and the attestation check run
//! here over a stream something else dialed. Each case runs the native
//! reqwest connector too, against the same shim, and the two must agree —
//! there is one implementation of what is checked.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use tinfoil_shim_mock::{Misbinding, RunningShim, ShimConfig};
use tinfoil_verifier::pipe::{AttestingPipe, TcpDialer};
use tinfoil_verifier::{AllowedMeasurement, AttestingClientConfig, Error, SevSnpPin};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MEASUREMENT: [u8; 48] = [0x5a; 48];
const UPSTREAM_BODY: &str = "from the upstream";

/// The one directory every shim's persistent roots live in.
fn cert_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let dir =
            std::env::temp_dir().join(format!("tinfoil-verifier-pipe-{}", std::process::id()));
        tinfoil_shim_mock::ensure_persistent_material(&dir).expect("shim roots");
        dir
    })
    .clone()
}

/// A plain-HTTP upstream answering every request with [`UPSTREAM_BODY`].
async fn upstream() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match socket.read(&mut byte).await {
                        Ok(1) => head.push(byte[0]),
                        _ => return,
                    }
                }
                let answer = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{UPSTREAM_BODY}",
                    UPSTREAM_BODY.len()
                );
                let _ = socket.write_all(answer.as_bytes()).await;
            });
        }
    });
    addr
}

async fn shim(misbind: Option<Misbinding>) -> RunningShim {
    let upstream = upstream().await;
    tinfoil_shim_mock::start(ShimConfig {
        upstream_url: format!("http://{upstream}"),
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        measurement: MEASUREMENT.to_vec(),
        host_data: [0; 32],
        cert_dir: cert_dir(),
        gpus: 0,
        extra_tls_names: Vec::new(),
        misbind,
    })
    .await
    .expect("start shim")
}

fn pin(measurement: [u8; 48]) -> AllowedMeasurement {
    AllowedMeasurement::sev_snp(SevSnpPin {
        measurement: hex::encode(measurement),
        host_data: hex::encode([0u8; 32]),
        guest_policy: format!("{:#x}", tinfoil_shim_mock::MOCK_GUEST_POLICY),
    })
}

fn base_url(shim: &RunningShim) -> String {
    format!("https://127.0.0.1:{}", shim.local_addr.port())
}

/// The configuration both transports are built from, with an observer
/// counting the attestations that passed.
fn config<'a>(
    shim: &'a RunningShim,
    pins: &'a [AllowedMeasurement],
    base_url: &'a str,
    passed: Arc<Mutex<usize>>,
) -> AttestingClientConfig<'a> {
    let mut tls_roots = rustls::RootCertStore::empty();
    tls_roots
        .add(rustls::pki_types::CertificateDer::from(
            shim.tls_ca_der.clone(),
        ))
        .unwrap();
    AttestingClientConfig {
        allowed_measurements: pins,
        inference_base_url: base_url,
        tls_roots,
        trusted_ark_der: Some(&shim.ark_der),
        trusted_ask_der: Some(&shim.ask_der),
        snp_min_tcb: None,
        snp_observer: None,
        attestation_observer: Some(Arc::new(move |_| *passed.lock().unwrap() += 1)),
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Attest over the pipe, then send one request on the attested stream:
/// `Ok` with what came back, or the check's refusal.
async fn over_pipe(
    shim: &RunningShim,
    pins: &[AllowedMeasurement],
    passed: Arc<Mutex<usize>>,
) -> Result<String, Error> {
    let base_url = base_url(shim);
    let pipe = AttestingPipe::new(config(shim, pins, &base_url, passed))?;
    let handshaken = pipe
        .handshake(&TcpDialer, "127.0.0.1", shim.local_addr.port())
        .await?;
    let mut stream = handshaken.attest(now()).await?;
    stream
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut answer = Vec::new();
    let _ = stream.read_to_end(&mut answer).await;
    Ok(String::from_utf8_lossy(&answer).into_owned())
}

/// The same request through the native connector: `Ok` with the body, or
/// the error text with its causes.
async fn over_reqwest(
    shim: &RunningShim,
    pins: &[AllowedMeasurement],
    passed: Arc<Mutex<usize>>,
) -> Result<String, String> {
    let base_url = base_url(shim);
    let client = tinfoil_verifier::attesting_client(config(shim, pins, &base_url, passed))
        .await
        .map_err(|e| e.to_string())?;
    match client.get(format!("{base_url}/v1/models")).send().await {
        Ok(response) => Ok(response.text().await.unwrap()),
        Err(e) => {
            let mut text = e.to_string();
            let mut source = std::error::Error::source(&e);
            while let Some(cause) = source {
                text.push_str(&format!(": {cause}"));
                source = cause.source();
            }
            Err(text)
        }
    }
}

#[tokio::test]
async fn a_pinned_enclave_is_attested_and_the_stream_carries_the_request() {
    let shim = shim(None).await;
    let pins = [pin(MEASUREMENT)];
    let passed = Arc::new(Mutex::new(0));

    let answer = over_pipe(&shim, &pins, passed.clone())
        .await
        .expect("attested over the pipe");
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert!(answer.contains(UPSTREAM_BODY), "{answer}");
    assert_eq!(*passed.lock().unwrap(), 1);
    assert_eq!(shim.attestations(), 1);

    let native = over_reqwest(&shim, &pins, passed.clone()).await;
    assert_eq!(native.as_deref(), Ok(UPSTREAM_BODY));
    assert_eq!(*passed.lock().unwrap(), 2);
}

#[tokio::test]
async fn a_measurement_that_is_not_pinned_is_refused() {
    let shim = shim(None).await;
    let pins = [pin([0x11; 48])];
    let passed = Arc::new(Mutex::new(0));

    let err = over_pipe(&shim, &pins, passed.clone()).await.unwrap_err();
    assert!(matches!(err, Error::MeasurementMismatch { .. }), "{err}");
    let native = over_reqwest(&shim, &pins, passed.clone())
        .await
        .unwrap_err();
    assert!(native.contains(&err.to_string()), "{native}");
    assert_eq!(*passed.lock().unwrap(), 0);
}

#[tokio::test]
async fn a_document_endorsing_another_tls_key_is_refused() {
    let shim = shim(Some(Misbinding::TlsKey)).await;
    let pins = [pin(MEASUREMENT)];
    let passed = Arc::new(Mutex::new(0));

    let err = over_pipe(&shim, &pins, passed.clone()).await.unwrap_err();
    assert!(matches!(err, Error::FingerprintMismatch { .. }), "{err}");
    let native = over_reqwest(&shim, &pins, passed.clone())
        .await
        .unwrap_err();
    assert!(native.contains("TLS fingerprint mismatch"), "{native}");
    assert_eq!(*passed.lock().unwrap(), 0);
}

#[tokio::test]
async fn a_report_not_committing_to_the_document_is_refused() {
    let shim = shim(Some(Misbinding::ReportData)).await;
    let pins = [pin(MEASUREMENT)];
    let passed = Arc::new(Mutex::new(0));

    let err = over_pipe(&shim, &pins, passed.clone()).await.unwrap_err();
    assert!(matches!(err, Error::ReportDataMismatch { .. }), "{err}");
    let native = over_reqwest(&shim, &pins, passed.clone())
        .await
        .unwrap_err();
    assert!(native.contains("report_data mismatch"), "{native}");
    assert_eq!(*passed.lock().unwrap(), 0);
}

/// TLS is the pipe's, not the host's: a peer whose certificate does not
/// chain to the caller's roots is refused before anything is attested.
#[tokio::test]
async fn a_peer_outside_the_tls_roots_is_refused_at_the_handshake() {
    let shim = shim(None).await;
    let pins = [pin(MEASUREMENT)];
    let base_url = base_url(&shim);
    let mut config = config(&shim, &pins, &base_url, Arc::new(Mutex::new(0)));
    config.tls_roots = rustls::RootCertStore::empty();
    config
        .tls_roots
        .extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let err = AttestingPipe::new(config)
        .unwrap()
        .handshake(&TcpDialer, "127.0.0.1", shim.local_addr.port())
        .await
        .err()
        .expect("refused");
    assert!(matches!(err, Error::Connect(_)), "{err}");
    assert_eq!(shim.attestations(), 0);
}
