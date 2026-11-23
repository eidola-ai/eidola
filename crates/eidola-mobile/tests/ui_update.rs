//! The Updates screen's store (`ui::update_store::UpdateStore`) under gpui's
//! test app, over a host that counts what reaches the shell: **nothing is
//! installed until the approval**, a release whose claims changed waits for
//! the reader's acceptance too, an approval is spent once, and a release that
//! failed a check is a warning that offers nothing.
//!
//! The release is verified evidence by fiat (`PendingBundleUpdate::from_evidence`,
//! as `tests/update.rs` takes it); the staging itself is `tests/update.rs`'s.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::updater::mobile::row::bundle_digest;
use eidola_app_core::updater::mobile::{
    ActivatedBundle, BundleState, BundleUpdate, EMBEDDED, PendingBundleUpdate, ServingBundle,
};
use eidola_app_core::updater::{ReleaseCheckError, ReleaseEvidence, ReleaseIndex, ReleaseSummary};
use eidola_mobile::protocol::Hello;
use eidola_mobile::ui::host::Host;
use eidola_mobile::ui::update_store::{UpdateDisplay, UpdateStore};
use futures_util::future::LocalBoxFuture;
use gpui::{AppContext as _, Entity, TestAppContext};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// A verified release recording a one-file bundle.
fn pending() -> PendingBundleUpdate {
    let file = b"<!doctype html>".to_vec();
    let row = serde_json::json!({
        "type": "bundle",
        "platform": "webview/wasm32",
        "sha256": format!("sha256:{}", hex(b"container")),
        "digest": format!("sha256:{}", bundle_digest([("index.html", hex(&file).as_str())].into_iter())),
        "files": [{"path": "index.html", "size": file.len(), "sha256": format!("sha256:{}", hex(&file))}],
    });
    let manifest = serde_json::to_vec(&serde_json::json!({
        "schema_version": 6,
        "artifacts": {"eidola-mobile-bundle": row},
    }))
    .unwrap();
    let index: ReleaseIndex = serde_json::from_value(serde_json::json!({
        "schema_version": 2, "version": "0.0.9", "git_commit": "9c3a".to_string() + &"0".repeat(36),
        "git_tag": "v0.0.9", "released_at": "2026-11-23T00:00:00Z",
        "artifact_manifest": {"url": "https://example.invalid/m.json", "sigstore_bundle_url": "https://example.invalid/m.sigstore"},
        "human_attestations": [],
        "artifacts": {"eidola-mobile-bundle": {"url": "https://example.invalid/b.tar"}},
    }))
    .unwrap();
    let evidence = ReleaseEvidence {
        summary: ReleaseSummary {
            version: index.version.clone(),
            git_commit: index.git_commit.clone(),
            git_tag: index.git_tag.clone(),
            released_at: index.released_at.clone(),
            previous_release: None,
            attestations: vec![],
        },
        manifest_sha256: hex(&manifest),
        manifest_bytes: manifest,
        ci_identity: String::new(),
        ci_rekor_log_index: 0,
    };
    match PendingBundleUpdate::from_evidence(&index, evidence).unwrap() {
        BundleUpdate::Available(p) => *p,
        other => panic!("no bundle: {other:?}"),
    }
}

fn state(active: &str) -> BundleState {
    BundleState {
        serving: ServingBundle {
            version: EMBEDDED.into(),
            digest: Some("0".repeat(64)),
            confirmed: true,
        },
        active: active.into(),
        confirmed: active == EMBEDDED,
        previous: EMBEDDED.into(),
        failed: None,
        versions: vec![],
    }
}

/// A host that offers what `check` says and counts every install, revert
/// and reload.
struct CountingHost {
    hello: Hello,
    check: RefCell<Option<Result<(), ReleaseCheckError>>>,
    installs: Rc<Cell<u32>>,
    reloads: Rc<Cell<u32>>,
}

impl Host for CountingHost {
    fn hello(&self) -> &Hello {
        &self.hello
    }
    fn open_external(&self, _: String) -> LocalBoxFuture<'static, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    fn bundle_state(&self) -> LocalBoxFuture<'static, Result<BundleState, String>> {
        Box::pin(async { Ok(state(EMBEDDED)) })
    }
    fn check_update(
        &self,
        _: Rc<AppCore>,
    ) -> LocalBoxFuture<'static, Result<BundleUpdate, ReleaseCheckError>> {
        let outcome = self.check.borrow().clone().unwrap_or(Ok(()));
        Box::pin(async move { outcome.map(|()| BundleUpdate::Available(Box::new(pending()))) })
    }
    fn install(
        &self,
        _: Rc<AppCore>,
        pending: PendingBundleUpdate,
    ) -> LocalBoxFuture<'static, Result<ActivatedBundle, AppError>> {
        self.installs.set(self.installs.get() + 1);
        let version = pending.version_name().to_owned();
        let digest = pending.bundle_digest().to_owned();
        Box::pin(async move {
            Ok(ActivatedBundle {
                state: state(&version),
                version,
                digest,
            })
        })
    }
    fn revert(&self) -> LocalBoxFuture<'static, Result<BundleState, AppError>> {
        Box::pin(async { Ok(state(EMBEDDED)) })
    }
    fn reload(&self) {
        self.reloads.set(self.reloads.get() + 1);
    }
    fn bundle_text(&self, _: &'static str) -> LocalBoxFuture<'static, Result<String, String>> {
        Box::pin(async { Err("none".into()) })
    }
}

struct Fixture {
    store: Entity<UpdateStore>,
    installs: Rc<Cell<u32>>,
    reloads: Rc<Cell<u32>>,
    _dir: tempfile::TempDir,
}

fn fixture(cx: &mut TestAppContext, check: Result<(), ReleaseCheckError>) -> Fixture {
    cx.executor().allow_parking();
    let dir = tempfile::tempdir().unwrap();
    let core = Rc::new(AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).unwrap());
    let installs = Rc::new(Cell::new(0));
    let reloads = Rc::new(Cell::new(0));
    let host: Rc<dyn Host> = Rc::new(CountingHost {
        hello: Hello {
            contract: 2,
            platform: "ios".into(),
            shell: "test".into(),
            methods: vec![],
        },
        check: RefCell::new(Some(check)),
        installs: installs.clone(),
        reloads: reloads.clone(),
    });
    let store = cx.update(|cx| cx.new(|cx| UpdateStore::new(core, host, cx)));
    Fixture {
        store,
        installs,
        reloads,
        _dir: dir,
    }
}

fn display(f: &Fixture, cx: &mut TestAppContext) -> UpdateDisplay {
    f.store.read_with(cx, |s, _| s.display().clone())
}

#[gpui::test]
fn nothing_is_installed_until_the_approval(cx: &mut TestAppContext) {
    let f = fixture(cx, Ok(()));
    f.store.update(cx, |s, cx| s.check_now(cx));
    cx.run_until_parked();
    let UpdateDisplay::Available(offer) = display(&f, cx) else {
        panic!("not offered: {:?}", display(&f, cx));
    };
    assert_eq!(f.installs.get(), 0, "a check installed something");

    // The fixture's manifest attests none of the claims this build expects:
    // the approval waits for the reader's acceptance of the change.
    assert!(offer.claims_changed());
    f.store.update(cx, |s, cx| s.approve(cx));
    cx.run_until_parked();
    assert_eq!(f.installs.get(), 0, "approved over changed claims");
    assert!(matches!(display(&f, cx), UpdateDisplay::Available(_)));

    f.store.update(cx, |s, cx| s.accept_claims(cx));
    f.store.update(cx, |s, cx| s.approve(cx));
    cx.run_until_parked();
    assert_eq!(f.installs.get(), 1);
    assert_eq!(f.reloads.get(), 1, "the app restarts into the update");
    assert!(matches!(display(&f, cx), UpdateDisplay::Activated { version } if version == "v0.0.9"));

    // The approval is spent: another tap installs nothing.
    f.store.update(cx, |s, cx| s.approve(cx));
    cx.run_until_parked();
    assert_eq!(f.installs.get(), 1);
}

#[gpui::test]
fn a_release_that_failed_a_check_is_a_warning_and_offers_nothing(cx: &mut TestAppContext) {
    let f = fixture(
        cx,
        Err(ReleaseCheckError::Rejected(
            "the signature does not verify".into(),
        )),
    );
    f.store.update(cx, |s, cx| s.check_now(cx));
    cx.run_until_parked();
    assert!(matches!(
        display(&f, cx),
        UpdateDisplay::Unverifiable { .. }
    ));
    f.store.update(cx, |s, cx| {
        s.accept_claims(cx);
        s.approve(cx);
    });
    cx.run_until_parked();
    assert_eq!(f.installs.get(), 0);
}

#[gpui::test]
fn an_unreachable_feed_is_quiet(cx: &mut TestAppContext) {
    let f = fixture(cx, Err(ReleaseCheckError::Unreachable("offline".into())));
    f.store.update(cx, |s, cx| s.check_now(cx));
    cx.run_until_parked();
    assert!(matches!(display(&f, cx), UpdateDisplay::CheckFailed { .. }));
}
