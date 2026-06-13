//! Store-level behavior tests — the regression gate for the state-2 stores
//! refactor (`docs/architecture/state.md`).
//!
//! These build *backend-backed* stores (a real `AppCore` over tempdirs,
//! pointed at an unreachable base URL) so the store task machinery actually
//! engages, but they assert only the **synchronous** state transition a
//! refresh performs *before* its task runs (entering `Loading` with a live
//! task in the slot). They deliberately do **not** `run_until_parked`: the
//! gpui `TestAppContext` scheduler enforces single-threaded determinism and
//! would flag the `AppCore` tokio runtime's background work as
//! non-deterministic. The synchronous transition is exactly the structural
//! property each test is about — no network result is needed.
//!
//! Not parking is no longer enough by itself: since zed's `TestScheduler`
//! grew a cross-thread activity detector (any wake of a gpui task from a
//! foreign thread records a non-determinism error, raced against the test
//! body — the unreachable-URL tasks fail fast on the tokio side and their
//! completion wakes the store task's gpui future from a tokio worker).
//! `backed_stores` therefore calls `cx.executor().allow_parking()`, the
//! upstream idiom for tests that intentionally mix real OS threads; it
//! disables the detector without changing what these tests assert.

use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::changes::Change;
use gpui::{AppContext, TestAppContext};

use eidola_gui::stores::{self, SpacesStore, Stores};

/// A real `AppCore` over tempdirs with an unreachable base URL. Its async
/// methods would fail fast if driven, but these tests never park the
/// scheduler, so the runtime stays idle — they only exercise the synchronous
/// store transitions. Returns the keepalive `TempDir`.
fn test_core() -> (Arc<AppCore>, tempfile::TempDir) {
    // Idempotent crypto-provider install (mirrors what AppCore::new needs).
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    let core = AppCore::new(config_dir, data_dir);
    core.set_base_url("https://127.0.0.1:1/v1".into()).unwrap();
    (Arc::new(core), dir)
}

fn backed_stores(cx: &mut TestAppContext) -> (Stores, tempfile::TempDir) {
    // Declare the real tokio runtime to the test scheduler (see module docs).
    cx.executor().allow_parking();
    let (core, dir) = test_core();
    let stores = cx.update(|cx| Stores::for_test(core, cx));
    (stores, dir)
}

/// The wave-2 launch-order bug: the first window's model list never loaded
/// because a shared `busy` flag let an in-flight startup op (wallet recovery)
/// drop the model fetch. With one task slot per store and no shared flag, both
/// start concurrently — neither can starve the other.
///
/// Deterministic replay: drive the launch sequence (wallet recovery, then the
/// first window's models refresh) and assert the model list *started* loading
/// (its own live task) rather than being dropped. The `Loading` transition is
/// synchronous — set the moment `refresh` is called — so the assertion holds
/// without running the (unreachable) network task.
#[gpui::test]
fn launch_order_does_not_starve_models(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);

    // Launch sequence, in the order `lib.rs::run()` issues it: startup wallet
    // recovery first, then the first chat window triggers the models refresh.
    stores.wallet.update(cx, |s, cx| {
        s.refresh(cx);
        s.recover(cx, |_, _, _| {});
    });
    stores.models.update(cx, |s, cx| s.refresh(cx));

    // Both have live tasks. The old shared-busy bug would have dropped the
    // models refresh entirely (it would never start), leaving it `NotLoaded`.
    // This `Loading` (a live ModelsStore task, concurrent with the in-flight
    // wallet recovery) is the structural fix: there is no shared gate to drop
    // it.
    stores.models.read_with(cx, |m, _| {
        assert!(
            m.models().is_loading(),
            "the model list refresh must start (its own task slot), not be \
             starved by the in-flight wallet recovery"
        );
    });
    stores.wallet.read_with(cx, |w, _| {
        assert!(w.is_loading(), "wallet recovery is also live, concurrently")
    });
}

/// The bus bridge dispatch: a `Change::Wallet` must drive
/// `WalletStore::refresh` (and only the wallet store). Exercises the bridge's
/// routing logic via the `dispatch_change_for_test` seam — deterministic, no
/// dependence on the tokio→gpui plumbing's timing (which the running app
/// exercises). A `Lagged` (`None`) refreshes everything.
#[gpui::test]
fn bus_bridge_dispatches_wallet_change(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);

    // Idle to start: nothing refreshed yet.
    stores.wallet.read_with(cx, |w, _| assert!(!w.is_loading()));
    stores
        .account
        .read_with(cx, |a, _| assert!(!a.balances().is_loading()));

    // A wallet change routes only to the wallet store.
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Wallet), cx));
    stores.wallet.read_with(cx, |w, _| {
        assert!(
            w.is_loading(),
            "a Change::Wallet must drive WalletStore::refresh"
        );
    });
    stores.account.read_with(cx, |a, _| {
        assert!(
            !a.balances().is_loading(),
            "a Change::Wallet must NOT touch the account store"
        );
    });

    // A `Lagged` signal (None) refreshes everything — every store kicks a
    // fresh load.
    cx.update(|cx| stores::dispatch_change_for_test(&stores, None, cx));
    stores
        .account
        .read_with(cx, |a, _| assert!(a.balances().is_loading()));
    stores
        .models
        .read_with(cx, |m, _| assert!(m.models().is_loading()));
}

/// A `Change::Record` must reach open Record windows. No global store owns
/// the Record's rows (they are window-scoped reader state), so the bridge
/// routes the change to the `RecordStore` relay — bumping its epoch, which
/// observing Record windows compare against to mark themselves stale. The
/// bug this guards: the dispatch silently dropped `Change::Record`, so an
/// open Record window never learned the trail grew (codex finding, PR #179).
#[gpui::test]
fn bus_bridge_routes_record_change_to_record_store(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);

    assert_eq!(stores.record.read_with(cx, |r, _| r.epoch()), 0);

    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Record), cx));
    assert_eq!(
        stores.record.read_with(cx, |r, _| r.epoch()),
        1,
        "a Change::Record must bump the RecordStore epoch"
    );

    // A `Lagged` (refresh-everything) may have dropped a Record change, so
    // it must bump too.
    cx.update(|cx| stores::dispatch_change_for_test(&stores, None, cx));
    assert_eq!(
        stores.record.read_with(cx, |r, _| r.epoch()),
        2,
        "a Lagged signal must also reach the RecordStore"
    );
}

/// The space-entity registry's join-existing semantics: two `open` calls for
/// the same id return the *same* `Space` entity (so two windows on one space
/// share one transcript + streaming buffer — wave-2 bug 4), while a different
/// id and each `blank()` yield distinct entities.
#[gpui::test]
fn spaces_registry_joins_existing_and_blanks_are_distinct(cx: &mut TestAppContext) {
    let store = cx.update(|cx| cx.new(|_| SpacesStore::stub(Vec::new())));

    let (a1, a2, b, blank1, blank2) = store.update(cx, |s, cx| {
        (
            s.open("space-a".into(), cx),
            s.open("space-a".into(), cx),
            s.open("space-b".into(), cx),
            s.blank(cx),
            s.blank(cx),
        )
    });

    assert_eq!(
        a1.entity_id(),
        a2.entity_id(),
        "two opens of one id must join the same Space entity"
    );
    assert_ne!(
        a1.entity_id(),
        b.entity_id(),
        "distinct ids get distinct entities"
    );
    assert_ne!(
        blank1.entity_id(),
        blank2.entity_id(),
        "each blank (⌘N) is its own id-less space until adopted"
    );
    assert!(
        blank1.read_with(cx, |sp, _| sp.id().is_none()),
        "a blank space starts id-less"
    );
}

/// Supersede semantics: two back-to-back refreshes on the same slot. Replacing
/// the task field drops (cancels) the predecessor, so only one live task ever
/// owns the cell — keep-newest, no interleaving. Both calls leave the cell
/// `Loading` with a single live task; the cell never holds a stale value from
/// a cancelled predecessor.
#[gpui::test]
fn refresh_supersede_cancels_predecessor(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);

    // First refresh starts a task and enters Loading.
    stores.account.update(cx, |s, cx| s.refresh_balances(cx));
    stores
        .account
        .read_with(cx, |a, _| assert!(a.balances().is_loading()));

    // Second refresh replaces the slot — the predecessor's `Task` is dropped
    // (cancelled). The cell is still `Loading` with exactly one live task, and
    // crucially never carries a value (no stale write could have interleaved).
    stores.account.update(cx, |s, cx| s.refresh_balances(cx));
    stores.account.read_with(cx, |a, _| {
        assert!(
            a.balances().is_loading(),
            "the surviving (latest) task leaves the cell Loading"
        );
        assert!(
            a.balances().value().is_none(),
            "supersede is keep-newest — no cancelled predecessor can interleave a value"
        );
    });
}
