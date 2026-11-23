//! The mobile app's self-update (`updater::mobile`), driven through the real
//! updater code against a bundle store that keeps the shell contract's
//! semantics (`mobile/shell-contract.md` → Bundle store): staging, sealing
//! with the digest computed from the bytes stored, and provisional
//! activation with its trial, confirmation and fallback.
//!
//! Two halves, as the desktop's install suite is split:
//!
//! - **Verification** runs for real against the captured `v0.0.8` release
//!   (its genuine CI-signed manifest and a throwaway-key attestation, the
//!   fixtures `tests/updates_check.rs` uses), so `prepare` is shown to be
//!   `verify_release` and nothing less. That release predates the mobile
//!   bundle and its manifest is schema 1, outside the accepted set, so the
//!   furthest a genuine release gets is that refusal, which the bundle
//!   reader makes only after `verify_release` has passed.
//! - **Staging** starts from `PendingBundleUpdate::from_evidence` given a
//!   synthetic schema-6 manifest as verified evidence, the way the desktop
//!   install tests start from an `InstallPlan`: a manifest carrying a bundle
//!   row cannot be CI-signed outside a release.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use eidola_app_core::storage::HostFuture;
use eidola_app_core::updater::mobile::container::write_ustar;
use eidola_app_core::updater::mobile::row::bundle_digest;
use eidola_app_core::updater::mobile::{
    self, BundleInstallError, BundleReport, BundleState, BundleStore, BundleStoreError,
    BundleUpdate, EMBEDDED, PendingBundleUpdate, SealedBundle, ServingBundle, StoredBundle,
};
use eidola_app_core::updater::{
    Fetcher, ReleaseCheckError, ReleaseEvidence, ReleaseIndex, ReleasePins, ReleaseSummary,
};
use sha2::{Digest, Sha256};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn fetcher() -> Fetcher {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    Fetcher::Network(reqwest::Client::builder().build().unwrap().into())
}

// ---------------------------------------------------------------------------
// A bundle store with the contract's semantics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct Version {
    sealed: bool,
    files: BTreeMap<String, Vec<u8>>,
    digest: Option<String>,
}

#[derive(Debug)]
struct Shell {
    versions: BTreeMap<String, Version>,
    embedded_digest: String,
    active: String,
    confirmed: bool,
    tried: bool,
    previous: String,
    failed: Option<String>,
    serving: String,
    serving_confirmed: bool,
    /// Every call, in order, as `method version`.
    calls: Vec<String>,
    /// Flip a byte of every write (a shell that stores something else).
    corrupt_writes: bool,
}

/// The contract's bundle store, in memory. Clones share one shell.
#[derive(Clone)]
struct FakeStore(Arc<Mutex<Shell>>);

impl FakeStore {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Shell {
            versions: BTreeMap::new(),
            embedded_digest: "e".repeat(64),
            active: EMBEDDED.into(),
            confirmed: true,
            tried: false,
            previous: EMBEDDED.into(),
            failed: None,
            serving: EMBEDDED.into(),
            serving_confirmed: true,
            calls: Vec::new(),
            corrupt_writes: false,
        })))
    }

    fn calls(&self) -> Vec<String> {
        self.0.lock().unwrap().calls.clone()
    }

    /// The calls that change the store: everything but `state`.
    fn mutations(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .filter(|c| !c.starts_with("state"))
            .collect()
    }

    fn stored(&self, version: &str) -> Option<BTreeMap<String, Vec<u8>>> {
        self.0
            .lock()
            .unwrap()
            .versions
            .get(version)
            .map(|v| v.files.clone())
    }

    /// A main-frame document load (launch or reload): the contract's three
    /// rules, then serve `active`.
    fn load_document(&self) {
        let mut s = self.0.lock().unwrap();
        let s = &mut *s;
        if !s.confirmed && !s.tried {
            s.tried = true;
        } else if !s.confirmed && s.tried {
            s.failed = Some(s.active.clone());
            s.active = s.previous.clone();
            s.confirmed = true;
            s.previous = EMBEDDED.into();
        }
        if s.active != EMBEDDED
            && !s
                .versions
                .get(&s.active)
                .is_some_and(|v| v.sealed && v.files.contains_key("index.html"))
        {
            s.failed = Some(s.active.clone());
            s.active = EMBEDDED.into();
            s.previous = EMBEDDED.into();
            s.confirmed = true;
        }
        s.serving = s.active.clone();
        s.serving_confirmed = s.confirmed;
    }

    /// `bundle.confirm`, as the new bundle's boot calls it.
    fn confirm(&self) {
        let mut s = self.0.lock().unwrap();
        if s.serving == s.active && !s.confirmed {
            s.confirmed = true;
            s.serving_confirmed = true;
            s.previous = EMBEDDED.into();
        }
    }

    fn snapshot(s: &Shell) -> BundleState {
        let digest_of = |v: &str| {
            if v == EMBEDDED {
                s.embedded_digest.clone()
            } else {
                s.versions
                    .get(v)
                    .and_then(|v| v.digest.clone())
                    .unwrap_or_default()
            }
        };
        BundleState {
            serving: ServingBundle {
                version: s.serving.clone(),
                digest: Some(digest_of(&s.serving)),
                confirmed: s.serving_confirmed,
            },
            active: s.active.clone(),
            confirmed: s.confirmed,
            previous: s.previous.clone(),
            failed: s.failed.clone(),
            versions: s
                .versions
                .iter()
                .map(|(name, v)| StoredBundle {
                    version: name.clone(),
                    sealed: v.sealed,
                    digest: v.digest.clone(),
                })
                .collect(),
        }
    }

    fn apply<T>(
        &self,
        call: String,
        f: impl FnOnce(&mut Shell) -> Result<T, BundleStoreError>,
    ) -> Result<T, BundleStoreError> {
        let mut s = self.0.lock().unwrap();
        s.calls.push(call);
        f(&mut s)
    }
}

fn invalid(m: &str) -> BundleStoreError {
    BundleStoreError::InvalidState(m.into())
}

impl BundleStore for FakeStore {
    fn begin<'a>(&'a self, version: &'a str) -> HostFuture<'a, Result<(), BundleStoreError>> {
        let r = self.apply(format!("begin {version}"), |s| {
            if version == EMBEDDED || s.versions.get(version).is_some_and(|v| v.sealed) {
                return Err(invalid("sealed or embedded"));
            }
            s.versions.insert(version.into(), Version::default());
            Ok(())
        });
        Box::pin(async move { r })
    }

    fn write<'a>(
        &'a self,
        version: &'a str,
        path: &'a str,
        offset: u64,
        data: &'a [u8],
    ) -> HostFuture<'a, Result<(), BundleStoreError>> {
        let r = self.apply(format!("write {version} {path}"), |s| {
            let corrupt = s.corrupt_writes;
            let v = s
                .versions
                .get_mut(version)
                .filter(|v| !v.sealed)
                .ok_or_else(|| invalid("not staging"))?;
            let file = v.files.entry(path.into()).or_default();
            let end = offset as usize + data.len();
            if file.len() < end {
                file.resize(end, 0);
            }
            file[offset as usize..end].copy_from_slice(data);
            if corrupt && !data.is_empty() {
                file[offset as usize] ^= 1;
            }
            Ok(())
        });
        Box::pin(async move { r })
    }

    fn seal<'a>(
        &'a self,
        version: &'a str,
    ) -> HostFuture<'a, Result<SealedBundle, BundleStoreError>> {
        let r = self.apply(format!("seal {version}"), |s| {
            let v = s
                .versions
                .get_mut(version)
                .filter(|v| !v.sealed)
                .ok_or_else(|| invalid("not staging"))?;
            let hashes: Vec<(String, String)> =
                v.files.iter().map(|(p, b)| (p.clone(), hex(b))).collect();
            let digest = bundle_digest(hashes.iter().map(|(p, h)| (p.as_str(), h.as_str())));
            v.sealed = true;
            v.digest = Some(digest.clone());
            Ok(SealedBundle {
                digest,
                files: v.files.len() as u64,
                bytes: v.files.values().map(|b| b.len() as u64).sum(),
            })
        });
        Box::pin(async move { r })
    }

    fn activate<'a>(
        &'a self,
        version: &'a str,
    ) -> HostFuture<'a, Result<BundleState, BundleStoreError>> {
        let r = self.apply(format!("activate {version}"), |s| {
            if version == EMBEDDED {
                s.active = EMBEDDED.into();
                s.previous = EMBEDDED.into();
                s.confirmed = true;
                return Ok(Self::snapshot(s));
            }
            if !s.versions.get(version).is_some_and(|v| v.sealed) {
                return Err(invalid("not sealed"));
            }
            if s.confirmed {
                s.previous = s.active.clone();
            }
            s.active = version.into();
            s.confirmed = false;
            s.tried = false;
            Ok(Self::snapshot(s))
        });
        Box::pin(async move { r })
    }

    fn remove<'a>(&'a self, version: &'a str) -> HostFuture<'a, Result<(), BundleStoreError>> {
        let r = self.apply(format!("remove {version}"), |s| {
            if version == EMBEDDED
                || version == s.active
                || version == s.previous
                || version == s.serving
            {
                return Err(invalid("in use"));
            }
            s.versions
                .remove(version)
                .map(drop)
                .ok_or_else(|| BundleStoreError::NotFound(version.into()))
        });
        Box::pin(async move { r })
    }

    fn state(&self) -> HostFuture<'_, Result<BundleState, BundleStoreError>> {
        let r = self.apply("state".into(), |s| Ok(Self::snapshot(s)));
        Box::pin(async move { r })
    }
}

// ---------------------------------------------------------------------------
// A release carrying a bundle
// ---------------------------------------------------------------------------

const TAG: &str = "v0.0.9";
const CONTAINER: &str = "eidola-mobile-bundle.tar";

fn bundle_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "boot.js",
            b"import init from './eidola_mobile.js'; init();".to_vec(),
        ),
        (
            "eidola_mobile.js",
            b"export default async function init() {}".to_vec(),
        ),
        // Larger than one block, and not a multiple of one.
        (
            "eidola_mobile_bg.wasm",
            (0..70_000u32).map(|i| (i % 251) as u8).collect(),
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

fn borrowed<'a>(files: &'a [(&'static str, Vec<u8>)]) -> Vec<(&'static str, &'a [u8])> {
    files.iter().map(|(p, b)| (*p, b.as_slice())).collect()
}

/// The manifest's bundle row for `listed` files, recording `container`.
fn bundle_row(listed: &[(&str, Vec<u8>)], container: &[u8]) -> serde_json::Value {
    let mut listed: Vec<(&str, &Vec<u8>)> = listed.iter().map(|(p, b)| (*p, b)).collect();
    listed.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let hashes: Vec<(String, String)> = listed
        .iter()
        .map(|(p, b)| (p.to_string(), hex(b)))
        .collect();
    serde_json::json!({
        "type": "bundle",
        "platform": "webview/wasm32",
        "sha256": format!("sha256:{}", hex(container)),
        "digest": format!("sha256:{}", bundle_digest(hashes.iter().map(|(p, h)| (p.as_str(), h.as_str())))),
        "files": listed.iter().map(|(p, b)| serde_json::json!({
            "path": p, "size": b.len(), "sha256": format!("sha256:{}", hex(b)),
        })).collect::<Vec<_>>(),
    })
}

fn manifest(schema: u64, row: Option<serde_json::Value>) -> Vec<u8> {
    let mut artifacts = serde_json::json!({
        "eidola-server": {"digest": "sha256:cc", "platform": "linux/amd64", "type": "oci"},
    });
    if let Some(row) = row {
        artifacts["eidola-mobile-bundle"] = row;
    }
    serde_json::to_vec_pretty(&serde_json::json!({
        "schema_version": schema,
        "artifacts": artifacts,
    }))
    .unwrap()
}

fn release(server: &MockServer, indexed: bool) -> ReleaseIndex {
    let base = format!("{}/download/{TAG}", server.uri());
    let mut index = serde_json::json!({
        "schema_version": 2,
        "version": TAG.trim_start_matches('v'),
        "git_commit": "9c3a000000000000000000000000000000000009",
        "git_tag": TAG,
        "released_at": "2026-11-23T00:00:00Z",
        "artifact_manifest": {
            "url": format!("{base}/artifact-manifest.json"),
            "sigstore_bundle_url": format!("{base}/artifact-manifest.json.sigstore"),
        },
        "human_attestations": [],
        "artifacts": {},
    });
    if indexed {
        index["artifacts"]["eidola-mobile-bundle"] =
            serde_json::json!({"url": format!("{base}/{CONTAINER}")});
    }
    serde_json::from_value(index).unwrap()
}

/// `verify_release`'s output for `manifest_bytes`, as though it verified.
fn evidence(manifest_bytes: Vec<u8>) -> ReleaseEvidence {
    ReleaseEvidence {
        summary: ReleaseSummary {
            version: TAG.trim_start_matches('v').into(),
            git_commit: "9c3a000000000000000000000000000000000009".into(),
            git_tag: TAG.into(),
            released_at: "2026-11-23T00:00:00Z".into(),
            previous_release: None,
            attestations: vec![],
        },
        manifest_sha256: hex(&manifest_bytes),
        manifest_bytes,
        ci_identity: format!(
            "https://github.com/eidola-ai/eidola/.github/workflows/sign-release.yml@refs/tags/{TAG}"
        ),
        ci_rekor_log_index: 1,
    }
}

fn pending(release: &ReleaseIndex, manifest_bytes: Vec<u8>) -> PendingBundleUpdate {
    match PendingBundleUpdate::from_evidence(release, evidence(manifest_bytes)).unwrap() {
        BundleUpdate::Available(pending) => *pending,
        other => panic!("expected a bundle, got {other:?}"),
    }
}

async fn serve_container(server: &MockServer, container: Vec<u8>) {
    Mock::given(method("GET"))
        .and(path(format!("/download/{TAG}/{CONTAINER}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(container))
        .mount(server)
        .await;
}

async fn container_requests(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|r| r.url.path().ends_with(CONTAINER))
        .count()
}

/// A release whose manifest records `listed` and whose container holds
/// `packed`, served; and its pending update.
async fn release_serving(
    listed: &[(&'static str, Vec<u8>)],
    packed: &[(&'static str, Vec<u8>)],
) -> (MockServer, PendingBundleUpdate) {
    let server = MockServer::start().await;
    let container = write_ustar(&borrowed(packed)).unwrap();
    let manifest = manifest(6, Some(bundle_row(listed, &container)));
    serve_container(&server, container).await;
    let pending = pending(&release(&server, true), manifest);
    (server, pending)
}

// ---------------------------------------------------------------------------
// Approval, staging, activation
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn an_approved_bundle_is_checked_staged_sealed_and_activated() {
    let files = bundle_files();
    let (server, pending) = release_serving(&files, &files).await;
    let store = FakeStore::new();

    // Verified and presented: nothing downloaded, nothing asked of the shell.
    let digest = pending.bundle_digest().to_string();
    assert_eq!(pending.version_name(), TAG);
    // The bundle row reads as well formed to the claims comparison too (the
    // test manifest is otherwise partial, so other rows differ).
    assert!(
        pending
            .claims()
            .deltas
            .iter()
            .all(|d| d.key != "artifacts.eidola-mobile-bundle"),
        "{:#?}",
        pending.claims().deltas
    );
    assert_eq!(container_requests(&server).await, 0);
    assert!(store.calls().is_empty());

    let activated = pending.approve(&fetcher(), &store).await.unwrap();
    assert_eq!(activated.digest, digest);
    assert_eq!(activated.state.active, TAG);
    assert!(!activated.state.confirmed);

    let mut expected = vec!["state".to_string(), format!("begin {TAG}")];
    let mut paths: Vec<&str> = files.iter().map(|(p, _)| *p).collect();
    paths.sort();
    expected.extend(paths.iter().map(|p| format!("write {TAG} {p}")));
    expected.extend([format!("seal {TAG}"), format!("activate {TAG}")]);
    assert_eq!(store.calls(), expected);

    let stored = store.stored(TAG).unwrap();
    assert_eq!(stored.len(), files.len());
    for (path, bytes) in &files {
        assert_eq!(&stored[*path], bytes, "{path}");
    }

    // Activated, not yet loaded: reported as waiting for the reload.
    let report = BundleReport::from_state(&activated.state);
    assert_eq!(report.awaiting_reload.as_deref(), Some(TAG));

    // The reload boots it on trial, and its boot confirms it.
    store.load_document();
    let state = store.state().await.unwrap();
    assert_eq!(
        state.serving.digest.as_deref(),
        Some(digest.as_str()),
        "the shell serves the verified bytes"
    );
    assert!(BundleReport::from_state(&state).on_trial);
    store.confirm();
    let state = store.state().await.unwrap();
    assert!(state.confirmed && state.serving.confirmed);
    assert_eq!(state.previous, EMBEDDED);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bundle_that_never_confirms_is_reported_and_reverted() {
    let files = bundle_files();
    let (_server, pending) = release_serving(&files, &files).await;
    let store = FakeStore::new();
    pending.approve(&fetcher(), &store).await.unwrap();

    // Its trial: loaded, and never confirmed (it crashed, hung, was killed).
    store.load_document();
    let on_trial = BundleReport::from_state(&store.state().await.unwrap());
    assert!(
        on_trial.on_trial,
        "an unconfirmed bundle is reported: {on_trial:?}"
    );

    // The next launch falls back, and says which bundle failed.
    store.load_document();
    let state = store.state().await.unwrap();
    assert_eq!(state.serving.version, EMBEDDED);
    assert_eq!(state.failed.as_deref(), Some(TAG));
    let report = BundleReport::from_state(&state);
    assert!(!report.on_trial);
    assert_eq!(report.failed.as_deref(), Some(TAG));

    // And the same release is not staged again to fail the same way.
    let files = bundle_files();
    let (_server, again) = release_serving(&files, &files).await;
    let before = store.mutations().len();
    let err = again.approve(&fetcher(), &store).await.unwrap_err();
    assert!(
        matches!(err, BundleInstallError::PreviouslyFailed { .. }),
        "{err}"
    );
    assert_eq!(store.mutations().len(), before, "nothing staged");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_user_can_revert_an_activated_bundle_to_the_embedded_one() {
    let files = bundle_files();
    let (_server, pending) = release_serving(&files, &files).await;
    let store = FakeStore::new();
    pending.approve(&fetcher(), &store).await.unwrap();
    store.load_document();

    let state = mobile::revert_to_embedded(&store).await.unwrap();
    assert_eq!(state.active, EMBEDDED);
    assert!(state.confirmed);
    store.load_document();
    let state = store.state().await.unwrap();
    assert_eq!(state.serving.version, EMBEDDED);
    assert_eq!(
        state.failed, None,
        "a revert the user chose is not a failure"
    );
}

/// Approve `pending` and require it to fail before anything reached the
/// shell: no staging call, nothing stored, nothing activated.
async fn refused_before_staging(pending: PendingBundleUpdate) -> BundleInstallError {
    let store = FakeStore::new();
    let err = pending.approve(&fetcher(), &store).await.unwrap_err();
    assert!(
        store.mutations().is_empty(),
        "staged anyway: {:?}",
        store.calls()
    );
    assert_eq!(store.state().await.unwrap().active, EMBEDDED);
    err
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tampered_container_is_refused_before_anything_is_staged() {
    let files = bundle_files();
    let server = MockServer::start().await;
    let container = write_ustar(&borrowed(&files)).unwrap();
    let manifest = manifest(6, Some(bundle_row(&files, &container)));
    let mut tampered = container.clone();
    // A byte of the wasm, well inside the container.
    tampered[6 * 512] ^= 1;
    serve_container(&server, tampered).await;
    let err = refused_before_staging(pending(&release(&server, true), manifest)).await;
    assert!(
        matches!(err, BundleInstallError::ContainerHash { .. }),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_container_with_the_right_files_but_other_bytes_is_refused() {
    // The same files in another order: every file is right and the
    // container is not the one recorded. The container's hash is checked on
    // its own, not inferred from its members.
    let files = bundle_files();
    let server = MockServer::start().await;
    let container = write_ustar(&borrowed(&files)).unwrap();
    let manifest = manifest(6, Some(bundle_row(&files, &container)));
    let mut reordered = files.clone();
    reordered.reverse();
    serve_container(&server, write_ustar(&borrowed(&reordered)).unwrap()).await;
    let err = refused_before_staging(pending(&release(&server, true), manifest)).await;
    assert!(
        matches!(err, BundleInstallError::ContainerHash { .. }),
        "{err}"
    );
}

// The next three are manifests that do not describe their own container:
// the container hash matches what was served, and the listing is the
// genuine bundle's. Each file is still held to the listing.

#[tokio::test(flavor = "multi_thread")]
async fn a_tampered_file_is_refused_before_anything_is_staged() {
    let files = bundle_files();
    let mut packed = files.clone();
    packed[3].1[20] ^= 1; // index.html
    let (_server, pending) = release_serving(&files, &packed).await;
    let err = refused_before_staging(pending).await;
    assert!(
        matches!(err, BundleInstallError::FileHash { ref path, .. } if path == "index.html"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_file_of_another_length_is_refused_before_anything_is_staged() {
    let files = bundle_files();
    let mut packed = files.clone();
    packed[0].1.push(b' ');
    let (_server, pending) = release_serving(&files, &packed).await;
    let err = refused_before_staging(pending).await;
    assert!(
        matches!(err, BundleInstallError::FileSize { ref path, .. } if path == "boot.js"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_missing_file_is_refused_before_anything_is_staged() {
    let files = bundle_files();
    let packed: Vec<_> = files
        .iter()
        .filter(|(p, _)| *p != "eidola_mobile.js")
        .cloned()
        .collect();
    let (_server, pending) = release_serving(&files, &packed).await;
    let err = refused_before_staging(pending).await;
    assert!(
        matches!(err, BundleInstallError::MissingFile { ref path } if path == "eidola_mobile.js"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_extra_file_is_refused_before_anything_is_staged() {
    let files = bundle_files();
    let mut packed = files.clone();
    packed.push(("worker.js", b"self.onmessage = () => {};".to_vec()));
    let (_server, pending) = release_serving(&files, &packed).await;
    let err = refused_before_staging(pending).await;
    assert!(
        matches!(err, BundleInstallError::UnlistedFile { ref path } if path == "worker.js"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_container_that_is_not_ustar_is_refused_before_anything_is_staged() {
    let files = bundle_files();
    let server = MockServer::start().await;
    let junk = vec![0x42u8; 4096];
    let manifest = manifest(6, Some(bundle_row(&files, &junk)));
    serve_container(&server, junk).await;
    let err = refused_before_staging(pending(&release(&server, true), manifest)).await;
    assert!(matches!(err, BundleInstallError::Container(_)), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_container_that_cannot_be_downloaded_stages_nothing() {
    let files = bundle_files();
    let server = MockServer::start().await;
    let container = write_ustar(&borrowed(&files)).unwrap();
    let manifest = manifest(6, Some(bundle_row(&files, &container)));
    // Nothing served: a 404.
    let err = refused_before_staging(pending(&release(&server, true), manifest)).await;
    assert!(matches!(err, BundleInstallError::Download { .. }), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shell_that_stores_other_bytes_is_caught_and_its_copy_removed() {
    let files = bundle_files();
    let (_server, pending) = release_serving(&files, &files).await;
    let store = FakeStore::new();
    store.0.lock().unwrap().corrupt_writes = true;
    let err = pending.approve(&fetcher(), &store).await.unwrap_err();
    assert!(
        matches!(
            err,
            BundleInstallError::Sealed {
                field: "digest",
                ..
            }
        ),
        "{err}"
    );
    let state = store.state().await.unwrap();
    assert_eq!(state.active, EMBEDDED, "nothing activated");
    assert!(
        state.versions.is_empty(),
        "the bad copy is removed: {state:?}"
    );
    assert!(!store.calls().iter().any(|c| c.starts_with("activate")));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_copy_left_by_an_earlier_attempt_is_replaced_not_trusted() {
    let files = bundle_files();
    let (_server, pending) = release_serving(&files, &files).await;
    let store = FakeStore::new();
    // A staging the app was killed in the middle of, under this version.
    store.begin(TAG).await.unwrap();
    store.write(TAG, "index.html", 0, b"half").await.unwrap();

    pending.approve(&fetcher(), &store).await.unwrap();
    let calls = store.calls();
    let removed = calls
        .iter()
        .position(|c| c == &format!("remove {TAG}"))
        .unwrap();
    let begun = calls
        .iter()
        .rposition(|c| c == &format!("begin {TAG}"))
        .unwrap();
    assert!(removed < begun, "{calls:?}");
    assert_eq!(
        store.stored(TAG).unwrap()["index.html"],
        bundle_files()[3].1
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_approval_repeated_after_activation_stages_nothing_more() {
    let files = bundle_files();
    let (_server, first) = release_serving(&files, &files).await;
    let store = FakeStore::new();
    first.approve(&fetcher(), &store).await.unwrap();
    let before = store.mutations();

    let (server, second) = release_serving(&files, &files).await;
    let activated = second.approve(&fetcher(), &store).await.unwrap();
    assert_eq!(activated.state.active, TAG);
    assert_eq!(store.mutations(), before);
    assert_eq!(container_requests(&server).await, 0);
}

// ---------------------------------------------------------------------------
// What a verified manifest must hold
// ---------------------------------------------------------------------------

fn from_evidence(
    release: &ReleaseIndex,
    evidence: ReleaseEvidence,
) -> Result<BundleUpdate, String> {
    PendingBundleUpdate::from_evidence(release, evidence).map_err(|e| match e {
        ReleaseCheckError::Rejected(m) => m,
        other => panic!("expected a rejection, got {other:?}"),
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn a_manifest_the_attestations_do_not_name_is_refused() {
    let server = MockServer::start().await;
    let files = bundle_files();
    let container = write_ustar(&borrowed(&files)).unwrap();
    let mut evidence = evidence(manifest(6, Some(bundle_row(&files, &container))));
    // The bytes read are not the bytes the attestations swore to.
    evidence.manifest_sha256 = hex(b"another manifest");
    let err = from_evidence(&release(&server, true), evidence).unwrap_err();
    assert!(err.contains("attestations name"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn evidence_for_another_release_is_refused() {
    let server = MockServer::start().await;
    let files = bundle_files();
    let container = write_ustar(&borrowed(&files)).unwrap();
    let mut evidence = evidence(manifest(6, Some(bundle_row(&files, &container))));
    evidence.summary.git_tag = "v0.0.8".into();
    let err = from_evidence(&release(&server, true), evidence).unwrap_err();
    assert!(err.contains("not for the release"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_manifest_is_held_to_the_schema_it_declares() {
    let server = MockServer::start().await;
    let files = bundle_files();
    let container = write_ustar(&borrowed(&files)).unwrap();
    let row = bundle_row(&files, &container);

    // Older than the bundle: a real release this app has nothing to take
    // from.
    assert!(matches!(
        from_evidence(&release(&server, true), evidence(manifest(5, None))),
        Ok(BundleUpdate::Unbundled { .. })
    ));
    // The schema promises a bundle and the manifest has none.
    let err = from_evidence(&release(&server, true), evidence(manifest(6, None))).unwrap_err();
    assert!(err.contains("records no"), "{err}");
    // A bundle under a schema that does not record one.
    let err = from_evidence(
        &release(&server, true),
        evidence(manifest(5, Some(row.clone()))),
    )
    .unwrap_err();
    assert!(err.contains("does not have it"), "{err}");
    // A schema this build does not accept: refused, never read, whether or
    // not a row that looks right is there; and one with no schema at all.
    for schema in [1, 7, 99] {
        let err = from_evidence(
            &release(&server, true),
            evidence(manifest(schema, Some(row.clone()))),
        )
        .unwrap_err();
        assert!(err.contains("does not accept"), "schema {schema}: {err}");
    }
    let mut unversioned: serde_json::Value =
        serde_json::from_slice(&manifest(6, Some(row.clone()))).unwrap();
    unversioned
        .as_object_mut()
        .unwrap()
        .remove("schema_version");
    let err = from_evidence(
        &release(&server, true),
        evidence(serde_json::to_vec(&unversioned).unwrap()),
    )
    .unwrap_err();
    assert!(err.contains("does not accept"), "{err}");
    // A bundle row that is not well formed.
    let mut bad = row.clone();
    bad["digest"] = format!("sha256:{}", "0".repeat(64)).into();
    let err = from_evidence(&release(&server, true), evidence(manifest(6, Some(bad)))).unwrap_err();
    assert!(err.contains("digest"), "{err}");
    // A bundle the release offers no download for.
    let err =
        from_evidence(&release(&server, false), evidence(manifest(6, Some(row)))).unwrap_err();
    assert!(err.contains("indexes no download"), "{err}");
}

// ---------------------------------------------------------------------------
// prepare: the desktop's verification, for real
// ---------------------------------------------------------------------------

const FIXTURE_MANIFEST: &[u8] = include_bytes!("fixtures/v0.0.8/artifact-manifest.json");
const FIXTURE_BUNDLE: &[u8] = include_bytes!("fixtures/v0.0.8/artifact-manifest.json.sigstore");
const FIXTURE_ATTESTATION: &[u8] = include_bytes!("fixtures/v0.0.8/attestation-throwaway.json");
const FIXTURE_ATTESTATION_BUNDLE: &[u8] =
    include_bytes!("fixtures/v0.0.8/attestation-throwaway.bundle.json");
const THROWAWAY_FINGERPRINT: &str =
    "9f2103792a8007816eb963d1b284d139ac3cf69f3d6ab7cb93878bb654d70915";
const FIXTURE_CI_IDENTITY_PATTERN: &str =
    "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v*";

fn fixture_pins() -> ReleasePins {
    let mut pins = ReleasePins::embedded();
    pins.trusted_attestant_fingerprints = vec![THROWAWAY_FINGERPRINT.into()];
    pins.ci_identity_pattern = FIXTURE_CI_IDENTITY_PATTERN.into();
    pins
}

/// The captured v0.0.8 release, served as the latest, with `manifest` in
/// place of its own.
async fn serve_v0_0_8(manifest: &[u8]) -> MockServer {
    let server = MockServer::start().await;
    let base = format!("{}/download/v0.0.8", server.uri());
    let index = serde_json::json!({
        "schema_version": 1,
        "version": "0.0.8",
        "git_commit": "bb7773fafbaa90d40e7c7634ff7f690b54dd9be9",
        "git_tag": "v0.0.8",
        "released_at": "2026-05-31T23:53:54Z",
        "previous_release": {
            "version": "0.0.6",
            "git_commit": "7ba323817820a04948ffc70e78955201c96c733f",
        },
        "artifact_manifest": {
            "url": format!("{base}/artifact-manifest.json"),
            "sigstore_bundle_url": format!("{base}/artifact-manifest.json.sigstore"),
        },
        "human_attestations": [{
            "attestant_id": "throwaway",
            "url": format!("{base}/attestation-throwaway.json"),
            "bundle_url": format!("{base}/attestation-throwaway.bundle.json"),
        }],
    });
    Mock::given(method("GET"))
        .and(path("/latest/download/release.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(index))
        .mount(&server)
        .await;
    for (name, bytes) in [
        ("artifact-manifest.json", manifest),
        ("artifact-manifest.json.sigstore", FIXTURE_BUNDLE),
        ("attestation-throwaway.json", FIXTURE_ATTESTATION),
        (
            "attestation-throwaway.bundle.json",
            FIXTURE_ATTESTATION_BUNDLE,
        ),
    ] {
        Mock::given(method("GET"))
            .and(path(format!("/download/v0.0.8/{name}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.to_vec()))
            .mount(&server)
            .await;
    }
    server
}

#[tokio::test(flavor = "multi_thread")]
async fn prepare_verifies_a_real_release_before_reading_its_bundle() {
    let server = serve_v0_0_8(FIXTURE_MANIFEST).await;
    let err = mobile::prepare(&fetcher(), &server.uri(), "0.0.6", None, &fixture_pins())
        .await
        .unwrap_err();
    // Verified for real (the signatures and the attestation all passed:
    // only the bundle reader, which runs after `verify_release`, says this),
    // and then refused: its manifest declares schema 1, which this build
    // does not accept. prepare has no way to stage anything regardless.
    assert!(
        matches!(&err, ReleaseCheckError::Rejected(m)
            if m.contains("declares schema_version 1, which this build does not accept")),
        "{err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn prepare_refuses_a_manifest_its_signatures_do_not_cover() {
    // A manifest that is not the one CI signed and the attestation names.
    let mut tampered = FIXTURE_MANIFEST.to_vec();
    let at = tampered.iter().position(|b| *b == b'{').unwrap() + 1;
    tampered.insert(at, b' ');
    let server = serve_v0_0_8(&tampered).await;
    let err = mobile::prepare(&fetcher(), &server.uri(), "0.0.6", None, &fixture_pins())
        .await
        .unwrap_err();
    assert!(matches!(err, ReleaseCheckError::Rejected(_)), "{err:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn prepare_refuses_a_release_attested_by_nobody_it_trusts() {
    let server = serve_v0_0_8(FIXTURE_MANIFEST).await;
    let mut pins = fixture_pins();
    pins.trusted_attestant_fingerprints = vec!["0".repeat(64)];
    let err = mobile::prepare(&fetcher(), &server.uri(), "0.0.6", None, &pins)
        .await
        .unwrap_err();
    assert!(matches!(err, ReleaseCheckError::Rejected(_)), "{err:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn prepare_offers_nothing_when_the_bundle_is_current() {
    let server = serve_v0_0_8(FIXTURE_MANIFEST).await;
    let prepared = mobile::prepare(&fetcher(), &server.uri(), "0.0.8", None, &fixture_pins())
        .await
        .unwrap();
    assert!(matches!(
        prepared,
        BundleUpdate::UpToDate { latest_version: Some(ref v) } if v == "0.0.8"
    ));
}
