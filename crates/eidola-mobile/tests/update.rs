//! The self-update on a shell: app-core's updater staging through the
//! contract's `bundle.*` calls (`update::ShellBundles`) against a shell
//! written in Rust (`fake_shell`) that keeps the contract's store and
//! provisional activation, and the new bundle's boot confirming it.
//!
//! The release is verified evidence by fiat
//! (`PendingBundleUpdate::from_evidence`), as in app-core's
//! `tests/updater_mobile.rs`, whose `prepare` tests run the verification
//! itself for real.

mod fake_shell;

use eidola_app_core::updater::mobile::container::write_ustar;
use eidola_app_core::updater::mobile::row::bundle_digest;
use eidola_app_core::updater::mobile::{BundleUpdate, PendingBundleUpdate};
use eidola_app_core::updater::{Fetcher, ReleaseEvidence, ReleaseIndex, ReleaseSummary};
use eidola_mobile::boot::{self, BootOptions, Booted};
use eidola_mobile::client::ShellClient;
use eidola_mobile::protocol::MAX_REQUEST_BYTES;
use eidola_mobile::update::{self, ShellBundles};
use fake_shell::{EMBEDDED_DIGEST, FakeShell};
use sha2::{Digest, Sha256};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TAG: &str = "v0.0.9";
const CONTAINER: &str = "eidola-mobile-bundle.tar";

fn hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

/// As `tests/bridge.rs`: outside any runtime, parking between polls.
fn park_on<F: std::future::Future>(f: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Unpark(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(output) = f.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}

fn boot_on(shell: &FakeShell) -> Booted<FakeShell> {
    let dir = std::env::temp_dir().join(format!(
        "eidola-mobile-update-{}-{:p}",
        std::process::id(),
        std::sync::Arc::as_ptr(&shell.files_key())
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let options = BootOptions {
        profile: Some(dir),
        ..BootOptions::default()
    };
    park_on(boot::boot(ShellClient::new(shell.clone()), options)).unwrap()
}

/// The bundle a release carries: its wasm larger than one `bundle.write`
/// may carry, so staging splits it.
fn bundle_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "boot.js",
            b"import init from './eidola_mobile.js'; init();".to_vec(),
        ),
        (
            "eidola_mobile_bg.wasm",
            (0..MAX_REQUEST_BYTES + 4096)
                .map(|i| (i % 253) as u8)
                .collect(),
        ),
        (
            "index.html",
            b"<!doctype html><script type=module src=boot.js></script>".to_vec(),
        ),
        (
            "snippets/eidola-mobile-0/js/bridge.js",
            b"export function platform() {}".to_vec(),
        ),
    ]
}

fn manifest(files: &[(&str, Vec<u8>)], container: &[u8]) -> Vec<u8> {
    let hashes: Vec<(String, String)> =
        files.iter().map(|(p, b)| (p.to_string(), hex(b))).collect();
    let row = serde_json::json!({
        "type": "bundle",
        "platform": "webview/wasm32",
        "sha256": format!("sha256:{}", hex(container)),
        "digest": format!("sha256:{}", bundle_digest(hashes.iter().map(|(p, h)| (p.as_str(), h.as_str())))),
        "files": files.iter().map(|(p, b)| serde_json::json!({
            "path": p, "size": b.len(), "sha256": format!("sha256:{}", hex(b)),
        })).collect::<Vec<_>>(),
    });
    serde_json::to_vec(&serde_json::json!({
        "schema_version": 6,
        "artifacts": {"eidola-mobile-bundle": row},
    }))
    .unwrap()
}

/// A release recording `files`, its container holding `packed`, served by
/// `server`; and its update, verified evidence by fiat.
async fn release(
    server: &MockServer,
    files: &[(&'static str, Vec<u8>)],
    packed: &[(&'static str, Vec<u8>)],
) -> PendingBundleUpdate {
    let packed: Vec<(&str, &[u8])> = packed.iter().map(|(p, b)| (*p, b.as_slice())).collect();
    let container = write_ustar(&packed).unwrap();
    let manifest_bytes = manifest(files, &container);
    Mock::given(method("GET"))
        .and(path(format!("/download/{TAG}/{CONTAINER}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(container))
        .mount(server)
        .await;
    let base = format!("{}/download/{TAG}", server.uri());
    let index: ReleaseIndex = serde_json::from_value(serde_json::json!({
        "schema_version": 2,
        "version": "0.0.9",
        "git_commit": "9c3a000000000000000000000000000000000009",
        "git_tag": TAG,
        "released_at": "2026-11-23T00:00:00Z",
        "artifact_manifest": {
            "url": format!("{base}/artifact-manifest.json"),
            "sigstore_bundle_url": format!("{base}/artifact-manifest.json.sigstore"),
        },
        "human_attestations": [],
        "artifacts": {"eidola-mobile-bundle": {"url": format!("{base}/{CONTAINER}")}},
    }))
    .unwrap();
    let evidence = ReleaseEvidence {
        summary: ReleaseSummary {
            version: "0.0.9".into(),
            git_commit: "9c3a000000000000000000000000000000000009".into(),
            git_tag: TAG.into(),
            released_at: "2026-11-23T00:00:00Z".into(),
            previous_release: None,
            attestations: vec![],
        },
        manifest_sha256: hex(&manifest_bytes),
        manifest_bytes,
        ci_identity: String::new(),
        ci_rekor_log_index: 0,
    };
    match PendingBundleUpdate::from_evidence(&index, evidence).unwrap() {
        BundleUpdate::Available(pending) => *pending,
        other => panic!("expected a bundle, got {other:?}"),
    }
}

fn fetcher() -> Fetcher {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    Fetcher::Network(reqwest::Client::builder().build().unwrap().into())
}

/// The store calls the shell received, in order (arrivals only).
fn bundle_calls(shell: &FakeShell) -> Vec<String> {
    shell
        .log()
        .into_iter()
        .filter_map(|l| l.strip_prefix("-> bundle.").map(str::to_owned))
        .collect()
}

#[test]
fn an_approved_update_is_staged_on_the_shell_and_confirmed_by_its_boot() {
    let shell = FakeShell::default();
    drop(boot_on(&shell)); // the embedded bundle, confirmed
    let client = ShellClient::new(shell.clone());
    let files = bundle_files();

    let digest = block_on(async {
        let server = MockServer::start().await;
        let pending = release(&server, &files, &files).await;
        let digest = pending.bundle_digest().to_string();
        // Verified and waiting for the user: the shell has been asked
        // nothing about bundles beyond the boot's confirm.
        assert_eq!(bundle_calls(&shell), ["confirm"]);

        let activated = pending
            .approve(&fetcher(), &ShellBundles::new(client.clone()))
            .await
            .unwrap();
        assert_eq!(activated.state.active, TAG);
        assert!(!activated.state.confirmed);
        let report = update::report(&client).await.unwrap();
        assert_eq!(report.awaiting_reload.as_deref(), Some(TAG));
        digest
    });

    // One begin, every file written (the wasm in two requests), sealed, then
    // activated, and nothing between.
    let wasm_writes = 2;
    let mut expected = vec![
        "confirm".to_string(),
        "state".into(),
        format!("begin {TAG}"),
    ];
    expected.extend(std::iter::repeat_n(
        format!("write {TAG}"),
        files.len() - 1 + wasm_writes,
    ));
    expected.extend([
        format!("seal {TAG}"),
        format!("activate {TAG}"),
        "state".into(),
    ]);
    assert_eq!(bundle_calls(&shell), expected);
    let stored = shell.bundles().versions[TAG].files.clone();
    for (path, bytes) in &files {
        assert_eq!(&stored[*path], bytes, "{path}");
    }

    // The reload: the shell boots it on trial, and its boot confirms it.
    let relaunched = shell.relaunch();
    let booted = boot_on(&relaunched);
    assert_eq!(booted.bundle.serving.version, TAG);
    assert_eq!(
        booted.bundle.serving.digest.as_deref(),
        Some(digest.as_str()),
        "the shell's own digest of what it serves is the release's"
    );
    assert!(booted.bundle.confirmed && booted.bundle.serving.confirmed);
    assert_eq!(booted.bundle.previous, "embedded");
}

#[test]
fn a_bundle_that_never_confirms_is_reverted_on_the_next_launch() {
    let shell = FakeShell::default();
    let client = ShellClient::new(shell.clone());
    let files = bundle_files();
    block_on(async {
        let server = MockServer::start().await;
        let pending = release(&server, &files, &files).await;
        pending
            .approve(&fetcher(), &ShellBundles::new(client.clone()))
            .await
            .unwrap();
    });

    // Its trial: the bundle loads and dies before it confirms.
    let trial = shell.relaunch();
    let state = block_on(ShellClient::new(trial.clone()).bundle_state()).unwrap();
    assert_eq!(state.serving.version, TAG);
    assert!(!state.serving.confirmed);
    let report = block_on(update::report(&ShellClient::new(trial.clone()))).unwrap();
    assert!(
        report.on_trial,
        "an unconfirmed bundle is reported: {report:?}"
    );

    // The next launch falls back to the embedded bundle and says why.
    let next = trial.relaunch();
    let booted = boot_on(&next);
    assert_eq!(booted.bundle.serving.version, "embedded");
    assert_eq!(
        booted.bundle.serving.digest.as_deref(),
        Some(EMBEDDED_DIGEST)
    );
    assert_eq!(booted.bundle.failed.as_deref(), Some(TAG));
    let report = block_on(update::report(&booted.client)).unwrap();
    assert_eq!(report.failed.as_deref(), Some(TAG));
}

#[test]
fn the_user_can_go_back_to_the_embedded_bundle() {
    let shell = FakeShell::default();
    let client = ShellClient::new(shell.clone());
    let files = bundle_files();
    block_on(async {
        let server = MockServer::start().await;
        let pending = release(&server, &files, &files).await;
        pending
            .approve(&fetcher(), &ShellBundles::new(client.clone()))
            .await
            .unwrap();
    });
    let running = shell.relaunch();
    drop(boot_on(&running));
    let state = park_on(update::revert(&ShellClient::new(running.clone()))).unwrap();
    assert_eq!(state.active, "embedded");
    let booted = boot_on(&running.relaunch());
    assert_eq!(booted.bundle.serving.version, "embedded");
    assert_eq!(booted.bundle.failed, None);
}

#[test]
fn a_tampered_bundle_never_reaches_the_shell() {
    let shell = FakeShell::default();
    let client = ShellClient::new(shell.clone());
    let files = bundle_files();
    let mut packed = files.clone();
    packed[2].1[0] ^= 1; // index.html
    let err = block_on(async {
        let server = MockServer::start().await;
        let pending = release(&server, &files, &packed).await;
        pending
            .approve(&fetcher(), &ShellBundles::new(client.clone()))
            .await
            .unwrap_err()
    });
    assert!(err.to_string().contains("index.html"), "{err}");
    assert_eq!(bundle_calls(&shell), ["state"], "only read, never staged");
    assert!(shell.bundles().versions.is_empty());
}
