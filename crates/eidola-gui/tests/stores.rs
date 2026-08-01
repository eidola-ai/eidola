//! Store-level behavior tests — the regression gate for the state-2 stores
//! refactor (`crates/eidola-gui/STATE.md`).
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
    let core = AppCore::new(config_dir, data_dir).expect("open core");
    core.runtime()
        .block_on(core.set_base_url("https://127.0.0.1:1/v1".into()))
        .unwrap();
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

/// `Change::Backends` routes to the registry snapshot *and* the model
/// catalogs (the set of destinations changed, so the catalog set is stale).
/// As everywhere in this file, only the synchronous `Loading` transition is
/// asserted — the tasks' results need no network.
#[gpui::test]
fn bus_bridge_routes_backends_change(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);

    stores.backends.read_with(cx, |b, _| {
        assert!(
            matches!(b.state(), eidola_gui::loadable::Loadable::NotLoaded),
            "backends start NotLoaded"
        );
    });

    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Backends), cx));

    stores.backends.read_with(cx, |b, _| {
        assert!(
            b.state().is_loading(),
            "a Change::Backends must start the registry refresh"
        );
    });
    stores.models.read_with(cx, |m, _| {
        assert!(
            m.models().is_loading(),
            "a Change::Backends must re-fetch the model catalogs"
        );
    });
}

/// A failed backend operation must reconcile its optimistic edit. `remove`
/// drops the row from the cached list immediately; when the core write
/// fails (the eidola singleton is built in and can't be removed) no
/// `Change::Backends` is emitted, so the failure arm itself re-fetches the
/// registry — otherwise the UI would keep showing durably-false state
/// (codex finding, PR #216). Unlike the transition-only tests above, this
/// one drives the (purely local-DB) tasks to completion by polling.
#[gpui::test]
fn backends_op_failure_refreshes_registry(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);

    // Load the real registry (a local DB read — no network involved).
    stores.backends.update(cx, |b, cx| b.refresh(cx));
    wait_for_backends(cx, &stores, "registry loads", |b| b.get("eidola").is_some());

    // The optimistic removal drops the row synchronously…
    stores
        .backends
        .update(cx, |b, cx| b.remove("eidola".into(), cx));
    stores.backends.read_with(cx, |b, _| {
        assert!(b.get("eidola").is_none(), "optimistic removal applies");
    });

    // …the core refuses, and the failure arm's refresh restores the row.
    wait_for_backends(cx, &stores, "failure refresh reconciles", |b| {
        b.get("eidola").is_some()
    });
    stores.backends.read_with(cx, |b, _| {
        assert!(b.op_error().is_some(), "the failure surfaces in op_error");
    });
}

/// Poll the backends store until `pred` holds (the tokio side is a local DB
/// op, so this settles in milliseconds; ~10s ceiling).
fn wait_for_backends(
    cx: &mut TestAppContext,
    stores: &Stores,
    what: &str,
    pred: impl Fn(&eidola_gui::stores::BackendsStore) -> bool,
) {
    for _ in 0..400 {
        cx.run_until_parked();
        if stores.backends.read_with(cx, |b, _| pred(b)) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    panic!("timed out waiting until {what}");
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

/// A mutation cancels any in-flight transcript load when it is accepted
/// (`supersede_load_for_mutation`) on the promise that its own post-commit
/// reload re-establishes the durable truth. The *failure* completion must
/// keep that promise too — the cancelled load may have carried another
/// writer's change, and a failed mutation can itself commit durable rows
/// whose `Change::Space` the runner-occupied bus guard dropped. The codex
/// finding on PR #206: the error arms only cleared the runner and emitted
/// `Failed`, stranding the space on a stale transcript until an unrelated
/// invalidation.
///
/// Backend-backed (a real `AppCore`, so `load_transcript` actually spawns),
/// asserting only the synchronous slot transitions per this module's rules;
/// the failure completion is replayed via the seam that delegates to the
/// production `fail_mutation`.
#[gpui::test]
fn space_mutation_failure_restarts_superseded_load(cx: &mut TestAppContext) {
    use eidola_app_core::error::AppError;

    let (stores, _dir) = backed_stores(cx);

    // Opening an existing space kicks the initial transcript load — a live
    // task in the load slot (this also stands in for a bus-driven refresh,
    // which occupies the same slot).
    let space = stores
        .spaces
        .update(cx, |s, cx| s.open("space-x".into(), cx));
    space.read_with(cx, |s, _| {
        assert!(
            s.has_pending_load_for_test(),
            "open() must start the initial transcript load"
        );
    });

    // Accepting a regenerate supersedes that load (the mutation prologue)...
    space.update(cx, |s, cx| {
        assert!(s.regenerate_post("a1".into(), "gemma4-31b".into(), cx));
        assert!(
            !s.has_pending_load_for_test(),
            "accepting a mutation cancels the in-flight load"
        );
    });

    // ...so the mutation *failing* must restart it.
    space.update(cx, |s, cx| {
        s.apply_chat_failure_for_test(
            AppError::Internal {
                message: "boom".into(),
            },
            cx,
        );
        assert!(
            s.has_pending_load_for_test(),
            "the failure completion must restart the transcript load it \
             cancelled at accept"
        );
        assert!(
            s.transcript().is_loading(),
            "the restarted load re-enters the in-flight state (every spinner \
             maps to a live task)"
        );
    });
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

/// The circadian appearance settings are synchronous config writes: the
/// `ConfigStore` writes through `AppCore` (persisting `config.toml`) and
/// re-reads the snapshot in the same update — no task, no bus round-trip
/// needed for the writing window. (Other windows converge via
/// `Change::Config`, exercised by `bus_bridge_dispatches_wallet_change`'s
/// sibling routing table.)
#[gpui::test]
fn config_store_circadian_settings_write_through(cx: &mut TestAppContext) {
    use eidola_app_core::config::{AppearanceSetting, LightCharacter, TimeOfDayTint};

    let (stores, _dir) = backed_stores(cx);

    stores.config.read_with(cx, |c, _| {
        let s = c.state().expect("backed store seeds a snapshot");
        // `auto` (follow the sun) is the shipped default since the
        // local-inference wave flipped it from `system`.
        assert_eq!(s.appearance, AppearanceSetting::Auto, "default");
        assert_eq!(s.time_of_day_tint, TimeOfDayTint::On, "default");
        assert_eq!(s.light_character, LightCharacter::Neutral, "default");
    });

    stores.config.update(cx, |c, cx| {
        c.set_appearance(AppearanceSetting::Day, cx);
        c.set_time_of_day_tint(TimeOfDayTint::Off, cx);
        c.set_light_character(LightCharacter::Warm, cx);
    });

    stores.config.read_with(cx, |c, _| {
        let s = c.state().expect("snapshot re-read after write");
        assert_eq!(s.appearance, AppearanceSetting::Day);
        assert_eq!(s.time_of_day_tint, TimeOfDayTint::Off);
        assert_eq!(s.light_character, LightCharacter::Warm);
    });
}

/// The View → Zoom In / Zoom Out / Actual Size ladder writes the `font_scale`
/// override through `AppCore` and re-reads the snapshot in the same update.
/// Stepping walks the ladder, saturates at the ends, and resets to Actual Size.
#[gpui::test]
fn config_store_zoom_ladder_writes_through(cx: &mut TestAppContext) {
    use eidola_app_core::config::{FONT_SCALE_DEFAULT, FONT_SCALE_MAX, FONT_SCALE_MIN};

    let (stores, _dir) = backed_stores(cx);

    // Fresh backed store opens at Actual Size.
    stores.config.read_with(cx, |c, _| {
        assert_eq!(c.font_scale(), FONT_SCALE_DEFAULT);
    });

    // Zoom In steps up one rung and persists it.
    stores.config.update(cx, |c, cx| c.zoom_in(cx));
    stores
        .config
        .read_with(cx, |c, _| assert_eq!(c.font_scale(), 1.1));

    // Zoom Out returns to the anchor.
    stores.config.update(cx, |c, cx| c.zoom_out(cx));
    stores
        .config
        .read_with(cx, |c, _| assert_eq!(c.font_scale(), FONT_SCALE_DEFAULT));

    // Zooming in past the top rung saturates at the max rather than overshooting.
    for _ in 0..12 {
        stores.config.update(cx, |c, cx| c.zoom_in(cx));
    }
    stores
        .config
        .read_with(cx, |c, _| assert_eq!(c.font_scale(), FONT_SCALE_MAX));

    // Actual Size resets from any zoom.
    stores.config.update(cx, |c, cx| c.reset_zoom(cx));
    stores
        .config
        .read_with(cx, |c, _| assert_eq!(c.font_scale(), FONT_SCALE_DEFAULT));

    // Zooming out past the bottom rung saturates at the min. (Persistence
    // across an AppCore re-open — the config.toml round-trip — is covered in
    // app-core's `set_circadian_settings_round_trip_through_config_state`.)
    for _ in 0..12 {
        stores.config.update(cx, |c, cx| c.zoom_out(cx));
    }
    stores
        .config
        .read_with(cx, |c, _| assert_eq!(c.font_scale(), FONT_SCALE_MIN));
}

/// A template write that carries a **router model** is two core calls
/// (`create_template` / `update_template`, then the dedicated
/// `set_template_router_model`), and the bus makes them race: app-core emits
/// `Change::Templates` from inside the first call, and dispatching it calls
/// `TemplatesStore::refresh` — which replaces the store's single task slot and
/// drops the gpui half of the op that is still in flight.
///
/// Split across two `bridge` calls, the second was only *constructed* after the
/// first await returned, so the cancellation swallowed it: the template was
/// created and its router silently left NULL. As one `bridge` closure — one
/// tokio future, whose `JoinHandle` `bridge` drops, so a dropped gpui receiver
/// cancels nothing core-side — both writes complete regardless.
///
/// Unlike its neighbours this test *does* park (it is about a genuinely
/// in-flight op, not a synchronous transition), and it asserts against the
/// **database** rather than the store snapshot: the cancelled relist means the
/// snapshot may predate the router write, which in production the setter's own
/// `Change::Templates` reconciles.
#[gpui::test]
fn template_router_write_survives_a_refresh_landing_mid_op(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    stores.templates.update(cx, |s, cx| {
        s.create(
            "Routed".into(),
            4,
            Vec::new(),
            Some("gemma4-31b".into()),
            cx,
        );
    });
    // Poll the op's future once: its tokio work is now running.
    cx.run_until_parked();
    // The bus event the first write already emitted, arriving mid-op.
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Templates), cx));

    // Read the durable state, not the (superseded) snapshot.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let listed = core
            .runtime()
            .block_on(core.list_space_templates())
            .expect("list templates");
        if let Some(t) = listed.iter().find(|t| t.title == "Routed") {
            if let Some(router) = t.router_model.as_deref() {
                assert_eq!(router, "gemma4-31b", "the router write landed");
                return;
            }
            // The template exists; give the second write its moment before
            // calling it lost.
            if std::time::Instant::now() > deadline {
                panic!("template created but its router model was dropped mid-op");
            }
        } else if std::time::Instant::now() > deadline {
            panic!("the create itself never landed");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}
