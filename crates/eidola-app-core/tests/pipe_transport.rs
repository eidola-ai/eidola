//! `AppCore` on the host-socket transport (`AppCore::with_host_network`), end
//! to end: every request is a stream a `TcpDialer` opens, with TLS and the
//! enclave attestation run in process, to an in-process `tinfoil-shim-mock`
//! fronting the chat harness's mock gateway. A turn — issuance, the spend,
//! the completion and its refund — passes only over attested connections, and
//! an endpoint whose attestation fails is refused before any request of the
//! turn is sent.

mod chat_harness;

use std::sync::Arc;

use base64::Engine as _;
use chat_harness::{MODEL, MockConfig, MockServer, with_account};
use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::transport::{HostNetwork, TcpDialer};
use tinfoil_shim_mock::{Misbinding, RunningShim, ShimConfig};

const MEASUREMENT: [u8; 48] = [0x6b; 48];

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

fn pem(der: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(der);
    let lines: Vec<&str> = b64
        .as_bytes()
        .chunks(64)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    format!(
        "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
        lines.join("\n")
    )
}

struct Stack {
    mock: MockServer,
    shim: RunningShim,
    core: AppCore,
    _dir: tempfile::TempDir,
}

/// The mock gateway behind a shim attesting [`MEASUREMENT`] (broken by
/// `misbind`), and a core on the host-socket transport that trusts
/// `trusted` — every value through the profile's own override surface, as
/// `eidola configure` writes it.
fn stack(misbind: Option<Misbinding>, trusted: [u8; 48]) -> Stack {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().expect("tempdir");
    let cert_dir = dir.path().join("certs");
    tinfoil_shim_mock::ensure_persistent_material(&cert_dir).expect("shim roots");

    // The host network needs the shim's TLS root before the core exists.
    let tls_ca = std::fs::read_to_string(cert_dir.join("tls-ca.pem")).expect("tls-ca.pem");
    let tls_ca_der = base64::engine::general_purpose::STANDARD
        .decode(
            tls_ca
                .lines()
                .filter(|line| !line.starts_with("-----"))
                .collect::<String>(),
        )
        .expect("tls-ca DER");
    let core = AppCore::with_host_network(
        dir.path().to_path_buf(),
        dir.path().join("data"),
        HostNetwork {
            dialer: Arc::new(TcpDialer),
            extra_tls_roots: vec![tls_ca_der.into()],
        },
    )
    .expect("open core");

    let (mock, shim) = core.runtime().block_on(async {
        let mock = chat_harness::start(MockConfig::default()).await;
        let shim = tinfoil_shim_mock::start(ShimConfig {
            upstream_url: mock.base_url.clone(),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            measurement: MEASUREMENT.to_vec(),
            host_data: [0; 32],
            cert_dir,
            gpus: 0,
            extra_tls_names: Vec::new(),
            misbind,
        })
        .await
        .expect("start shim");
        (mock, shim)
    });

    core.runtime().block_on(async {
        core.set_base_url(format!("https://127.0.0.1:{}", shim.local_addr.port()))
            .await
            .expect("base url");
        core.set_hardware_root_ca(pem(&shim.ark_der))
            .await
            .expect("ARK");
        core.set_hardware_intermediate_ca(pem(&shim.ask_der))
            .await
            .expect("ASK");
        core.trust_measurement(
            hex::encode(trusted),
            hex::encode([0u8; 32]),
            format!("{:#x}", tinfoil_shim_mock::MOCK_GUEST_POLICY),
        )
        .await
        .expect("trust measurement");
    });
    with_account(&core);
    Stack {
        mock,
        shim,
        core,
        _dir: dir,
    }
}

#[test]
fn a_turn_is_paid_for_and_answered_over_attested_connections() {
    run(|| {
        let stack = stack(None, MEASUREMENT);
        let result = stack
            .core
            .runtime()
            .block_on(stack.core.chat("over the pipe".into(), MODEL.into(), None))
            .expect("the turn succeeds over the attested pipe");
        assert_eq!(result.content, "Hello from the mock.");
        assert_eq!(stack.mock.chat_hits(), 1);
        // Every connection is attested on its own: at least the issuance
        // requests and the completion each opened one.
        assert!(
            stack.shim.attestations() >= 3,
            "{} attestations",
            stack.shim.attestations()
        );
        // The spend settled on the refund the answer carried.
        let wallet = stack
            .core
            .runtime()
            .block_on(stack.core.wallet_credentials())
            .expect("wallet");
        assert!(!wallet.is_empty(), "a successor credential remains");
    });
}

/// The refusals the attestation check makes reach the caller as attestation
/// failures, and nothing of the turn reaches the upstream.
fn refused(misbind: Option<Misbinding>, trusted: [u8; 48]) -> AppError {
    let stack = stack(misbind, trusted);
    let err = stack
        .core
        .runtime()
        .block_on(stack.core.chat("refused".into(), MODEL.into(), None))
        .expect_err("the turn must be refused");
    assert_eq!(stack.mock.chat_hits(), 0);
    assert!(
        stack
            .core
            .runtime()
            .block_on(stack.core.wallet_credentials())
            .expect("wallet")
            .is_empty(),
        "nothing was issued"
    );
    err
}

#[test]
fn a_measurement_that_is_not_trusted_is_refused() {
    run(|| {
        let err = refused(None, [0x11; 48]);
        assert!(
            matches!(&err, AppError::Attestation { message } if message.contains("trusted_measurements")),
            "{err:?}"
        );
    });
}

#[test]
fn a_document_bound_to_another_tls_key_is_refused() {
    run(|| {
        let err = refused(Some(Misbinding::TlsKey), MEASUREMENT);
        assert!(
            matches!(&err, AppError::Attestation { message } if message.contains("TLS certificate does not match")),
            "{err:?}"
        );
    });
}

#[test]
fn a_report_not_committing_to_the_document_is_refused() {
    run(|| {
        let err = refused(Some(Misbinding::ReportData), MEASUREMENT);
        assert!(
            matches!(&err, AppError::Attestation { message } if message.contains("report_data mismatch")),
            "{err:?}"
        );
    });
}
