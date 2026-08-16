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
use eidola_app_core::changes::{Change, ChangeOrigin};
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
/// id and each `create()` yield distinct entities — two ⌘N windows are two
/// conversations, each with its own id from birth.
#[gpui::test]
fn spaces_registry_joins_existing_and_new_spaces_are_distinct(cx: &mut TestAppContext) {
    let store = cx.update(|cx| cx.new(|_| SpacesStore::stub(Vec::new())));

    let (a1, a2, b, new1, new2) = store.update(cx, |s, cx| {
        (
            s.open("space-a".into(), cx),
            s.open("space-a".into(), cx),
            s.open("space-b".into(), cx),
            s.create(cx),
            s.create(cx),
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
        new1.entity_id(),
        new2.entity_id(),
        "each ⌘N is its own space"
    );
    let (id1, id2) = (
        new1.read_with(cx, |sp, _| sp.id().to_string()),
        new2.read_with(cx, |sp, _| sp.id().to_string()),
    );
    assert_ne!(id1, id2, "and each carries its own id from birth");
    // Registered under that id at once, which is what makes every store-wide
    // broadcast reach a brand-new space by ordinary membership.
    let rejoined = store.update(cx, |s, cx| s.open(id1, cx));
    assert_eq!(rejoined.entity_id(), new1.entity_id());
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

/// The quit path closes the bus bridge, and nothing dispatches after it.
///
/// This is the *class* cure for a bug that had already been patched twice at
/// the emitter: silencing app-core's quit-time engine drain fixed one, its
/// engine supervisors a second, and an in-flight model **download** reporting
/// progress during the quit's grace window would have been a third. Any
/// `Change` dispatched after `App::shutdown` sets `quitting` reaches a store's
/// `refresh` → `cx.spawn` → gpui's "Can't spawn on main thread after
/// on_app_quit" panic — so the fix belongs at the single door every `Change`
/// in the process comes through, not at each emitter in turn.
///
/// Drives the **live** tokio→gpui plumbing (the rest of this file uses the
/// `dispatch_change_for_test` seam), because closing the loop is exactly what
/// is being asserted.
#[gpui::test]
fn a_quiesced_bus_bridge_dispatches_nothing(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores have a core");
    let bridge = cx.update(|cx| stores::install_bus_bridge(&stores, cx));

    // A real write on the core emits `Change::SpaceIndex`; the bridge should
    // carry it into the SpacesStore without anyone asking.
    core.runtime()
        .block_on(core.create_space(Some("first".into())))
        .expect("create space");
    wait_until(cx, "the live bridge dispatches a change", |cx| {
        stores.spaces.read_with(cx, |s, _| s.list().len() == 1)
    });

    bridge.quiesce();
    assert!(bridge.is_quiesced());

    core.runtime()
        .block_on(core.create_space(Some("second".into())))
        .expect("create space");
    // Give the bridge every chance to dispatch it — the assertion is that it
    // does not.
    for _ in 0..8 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    assert_eq!(
        stores.spaces.read_with(cx, |s, _| s.list().len()),
        1,
        "a quiesced bridge must dispatch nothing, however the change arrives"
    );
}

/// A template write that carries a **router model** is two core calls
/// (`create_template` / `update_template`, then the dedicated
/// `set_template_router_model`), and the bus makes them race: app-core emits
/// `Change::Templates` from inside the first call, and dispatching it calls
/// `TemplatesStore::refresh`.
///
/// Split across two `bridge` calls, the second is only *constructed* after the
/// first await returns, so anything that replaces the op's slot in between
/// swallows it: the template is created and its router silently left NULL. As
/// one `bridge` closure — one tokio future, whose `JoinHandle` `bridge` drops,
/// so a dropped gpui receiver cancels nothing core-side — both writes complete
/// regardless. The refresh is no longer a superseder (mutations own their own
/// slot — see the store's docs), but another **mutation** still is, so the
/// one-future composition this drives stays the guard.
///
/// Unlike its neighbours this test *does* park (it is about a genuinely
/// in-flight op, not a synchronous transition), and it asserts against the
/// **database** rather than the store snapshot, which is what a superseded op
/// would have left behind.
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

/// Poll `run_until_parked` until `pred` holds — a write-then-relist round-trips
/// through the tokio runtime, which `run_until_parked` alone can return before.
fn wait_until(
    cx: &mut TestAppContext,
    what: &str,
    mut pred: impl FnMut(&mut TestAppContext) -> bool,
) {
    for _ in 0..400 {
        cx.run_until_parked();
        if pred(cx) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    panic!("timed out waiting until {what}");
}

/// **A bus-driven refresh must not cancel a mutation's completion.** Both
/// write-through stores compose `write; re-list` and report a failure in
/// `op_error` — and both are refreshed by signals they do not raise
/// (`Change::Templates`, which a write of its own emits *before it returns*;
/// `Change::Participants`, which any shared-participant edit anywhere emits).
/// While the refresh shared the mutation's task slot, one landing mid-write
/// replaced it and dropped the gpui half of the op: the core write still ran
/// (`bridge` drops the tokio `JoinHandle`, cancelling nothing), but the
/// continuation that surfaces the error and applies the re-list never did — so
/// a *refused* write was indistinguishable from a successful one.
///
/// Driven at the narrowest point there is: the refresh is dispatched before the
/// op's future has been polled even once.
#[gpui::test]
fn a_refresh_landing_mid_write_keeps_the_templates_ops_error(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    stores.templates.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the registry loads", |cx| {
        stores.templates.read_with(cx, |s, _| !s.list().is_empty())
    });

    // Make the write fail before it writes anything: the router reference's
    // backend is gone by the time the create runs.
    core.runtime()
        .block_on(core.set_backend_enabled("eidola".into(), false))
        .expect("disable eidola");
    stores.templates.update(cx, |s, cx| {
        s.create(
            "Doomed".into(),
            4,
            Vec::new(),
            Some("gemma4-31b".into()),
            cx,
        );
    });
    // An unrelated shared-participant edit, landing mid-write.
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx));

    wait_until(cx, "the refusal surfaces", |cx| {
        stores
            .templates
            .read_with(cx, |s, _| s.op_error().is_some())
    });
    // The same continuation carries the re-list, so the registry is a listing
    // and not whatever a cancelled op left behind — and the cell it took over
    // from the cancelled refresh is *resolved*, never left mid-flight (the
    // strand `stores::settle_mutation` exists to prevent; its own failure
    // quadrants are unit-tested there, since a DB read cannot be made to fail
    // through this seam).
    stores.templates.read_with(cx, |s, _| {
        assert!(
            s.list().iter().any(|t| t.title == "Default"),
            "the mutation re-listed on its failure exit"
        );
        assert!(
            !s.templates().is_loading(),
            "the mutation must resolve the cell it took the read from"
        );
    });
}

/// The per-space half of the rule above — and the sharper case, because
/// `ParticipantsStore`'s own writes emit the very `Change::Participants` that
/// drove `refresh_all` into their slot.
#[gpui::test]
fn a_refresh_landing_mid_write_keeps_the_participants_ops_error(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("create space")
        .id;

    stores
        .participants
        .update(cx, |s, cx| s.ensure(space.clone(), cx));
    wait_until(cx, "the roster loads", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.list(&space).is_empty())
    });

    // A label carrying a line break is refused before any write (it would
    // inject a second authored paragraph into the wire header).
    stores.participants.update(cx, |s, cx| {
        s.update_everywhere(
            space.clone(),
            eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            eidola_app_core::ParticipantUpdate {
                label: Some("You\nand me".into()),
                ..Default::default()
            },
            eidola_app_core::ExpectedScope::Any,
            cx,
        );
    });
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx));

    wait_until(cx, "the refusal surfaces", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.op_errors_for(&space).is_empty())
    });
    stores.participants.read_with(cx, |s, _| {
        assert!(
            !s.list(&space).is_empty(),
            "the mutation re-listed on its failure exit"
        );
        assert!(
            !s.participants(&space).is_loading(),
            "the mutation must resolve the cell it took the read from"
        );
    });
}

// ---------------------------------------------------------------------------
// SpaceSettingsStore — the space inspector's per-space domain cell.
// ---------------------------------------------------------------------------

/// The write-through round trip: each setter writes core-side and the store's
/// own re-read (not the bus) is what lands the new value, so the panel is
/// correct even on a bus-less run.
#[gpui::test]
fn space_settings_write_through_and_reread(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("create space")
        .id;

    stores
        .space_settings
        .update(cx, |s, cx| s.ensure(space.clone(), cx));
    wait_until(cx, "the settings load", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).has_value())
    });
    stores.space_settings.read_with(cx, |s, _| {
        assert_eq!(
            s.settings(&space).value().map(|v| v.cascade_limit),
            Some(eidola_app_core::DEFAULT_CASCADE_LIMIT)
        );
    });

    stores
        .space_settings
        .update(cx, |s, cx| s.set_cascade_limit(space.clone(), 7, cx));
    wait_until(cx, "the new limit lands", |cx| {
        stores.space_settings.read_with(cx, |s, _| {
            s.settings(&space).value().map(|v| v.cascade_limit) == Some(7)
        })
    });

    // Off round-trips as an ordinary choice, and a reference whose backend is
    // gone is refused into the op-error slot rather than silently dropped.
    stores.space_settings.update(cx, |s, cx| {
        s.set_router_model(space.clone(), Some("nothing@nope".into()), cx)
    });
    wait_until(cx, "the refusal surfaces", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.op_error(&space).is_some())
    });
    stores.space_settings.read_with(cx, |s, _| {
        assert_eq!(
            s.settings(&space).value().map(|v| v.router_model.clone()),
            Some(None),
            "a refused write leaves the setting where it was"
        );
        assert!(
            !s.settings(&space).is_loading(),
            "the mutation must resolve the cell it took the read from"
        );
    });
}

/// The refresh-vs-mutation rule for this store: its own write emits
/// `Change::Space`, which routes back here as a refresh — sharing one slot
/// would cancel the write's continuation and lose both its `op_error` and its
/// re-read.
#[gpui::test]
fn a_refresh_landing_mid_write_keeps_the_space_settings_op_error(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("create space")
        .id;

    stores
        .space_settings
        .update(cx, |s, cx| s.ensure(space.clone(), cx));
    wait_until(cx, "the settings load", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).has_value())
    });

    // Refused before it writes anything (the floor is 1), with the space's own
    // bus signal landing mid-write.
    stores
        .space_settings
        .update(cx, |s, cx| s.set_cascade_limit(space.clone(), 0, cx));
    cx.update(|cx| {
        stores::dispatch_change_for_test(&stores, Some(Change::Space(space.clone())), cx)
    });

    wait_until(cx, "the refusal surfaces", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.op_error(&space).is_some())
    });
    stores.space_settings.read_with(cx, |s, _| {
        assert!(
            s.settings(&space).has_value(),
            "the mutation re-read on its failure exit"
        );
        assert!(
            !s.settings(&space).is_loading(),
            "the mutation must resolve the cell it took the read from"
        );
    });
}

/// `Change::Space` fires on every post; only a space something is actually
/// looking at should pay for a re-read.
#[gpui::test]
fn a_space_change_re_reads_only_cached_settings(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);

    cx.update(|cx| {
        stores::dispatch_change_for_test(&stores, Some(Change::Space("unseen".into())), cx)
    });
    stores.space_settings.read_with(cx, |s, _| {
        assert!(
            !s.settings("unseen").is_loading(),
            "a space with no inspector open is never fetched"
        );
    });
}

// ---------------------------------------------------------------------------
// SpacesStore — the Library index's optimistic mutations.
// ---------------------------------------------------------------------------

/// A listing row the database does not have — the stale-id case (another writer
/// archived the space; a listing that outlived its rows). Seeded through the
/// store's own settle, so the cell is exactly what a landed re-list leaves.
fn stale_row(id: &str) -> eidola_app_core::SpaceInfo {
    let ts = eidola_app_core::now_ms();
    eidola_app_core::SpaceInfo {
        id: id.into(),
        title: Some("Tides".into()),
        snippet: None,
        created_at: ts,
        last_activity_at: ts,
        message_count: 2,
        archived_at: None,
    }
}

/// **A refused rename must not stand.** The optimistic title edit is what makes
/// the Library answer without a round trip; ignoring the write's `Err` left the
/// Library, the window title, and the inspector's title field all showing a name
/// nothing ever persisted, until an unrelated refresh took it away without a
/// word. The cure is the write-through shape: the refusal lands in `op_error`
/// (tagged with the space it was about) and the re-list runs on the failure
/// exit, reconciling the cached index back to what the database holds.
#[gpui::test]
fn a_refused_rename_reconciles_the_index_and_surfaces_the_refusal(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let real = core
        .runtime()
        .block_on(core.create_space(Some("Nile".into())))
        .expect("create space")
        .id;

    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(None, Ok(vec![stale_row("ghost")]), None, cx)
    });

    stores
        .spaces
        .update(cx, |s, cx| s.rename("ghost".into(), "Renamed".into(), cx));
    // The refusal and the reconcile are two events now: the write's refusal is
    // reported as it settles, and the batch-end read — issued only once no write
    // is in flight — lands after it.
    wait_until(cx, "the refusal surfaces and the index reconciles", |cx| {
        stores.spaces.read_with(cx, |s, _| {
            s.op_error().is_some() && !s.list().iter().any(|r| r.id == "ghost")
        })
    });
    stores.spaces.read_with(cx, |s, _| {
        assert!(
            s.op_error_for("ghost").is_some(),
            "the refusal is tagged with the space it was about, so only that \
             space's inspector shows it"
        );
        assert!(
            !s.list().iter().any(|r| r.id == "ghost"),
            "the optimistic title never stands: the batch-end read reconciles \
             on the failure exit"
        );
        assert!(
            s.list().iter().any(|r| r.id == real),
            "and what it lands is the database's listing"
        );
        assert!(
            !s.index().is_loading(),
            "the mutation must resolve the cell it took the read from"
        );
    });
}

/// The same rule for archive, over its one refusal that is not an error: a
/// `false` return means nothing was archived (the space was already archived —
/// the race the Library's × can lose — or is not there at all), so the row this
/// operation had already dropped from the listing was not its to drop. Say so
/// rather than let an optimistic removal read as an unearned success.
///
/// The teeth here are the *refusal*: this scenario's listing is empty whether or
/// not the re-list is conditional (the space really is archived). The re-list's
/// own teeth live on the rename above, where a refused write's optimistic edit
/// is what the re-list has to take back.
#[gpui::test]
fn an_archive_that_changed_nothing_says_so(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(Some("Tides".into())))
        .expect("create space")
        .id;

    stores.spaces.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the index loads", |cx| {
        stores
            .spaces
            .read_with(cx, |s, _| s.list().iter().any(|r| r.id == space))
    });

    // Another writer archives it first. (No bus bridge in these tests, so the
    // cached index still shows the row — exactly the state a second window is
    // in between the write and its invalidation.)
    core.runtime()
        .block_on(core.archive_space(space.clone()))
        .expect("archive");

    stores
        .spaces
        .update(cx, |s, cx| s.archive(space.clone(), cx));
    wait_until(cx, "the refusal surfaces and the index reconciles", |cx| {
        stores
            .spaces
            .read_with(cx, |s, _| s.op_error().is_some() && s.list().is_empty())
    });
    stores.spaces.read_with(cx, |s, _| {
        assert!(s.op_error_for(&space).is_some());
        assert!(s.list().is_empty(), "and the listing is re-read either way");
        assert!(!s.index().is_loading());
    });
}

/// The refresh-vs-mutation rule for this store — and the sharpest case there
/// is, because a rename emits the very `Change::SpaceIndex` that routes back
/// here as a refresh. While the two shared one slot, that refresh replaced the
/// write's task: the core write still ran (`bridge` drops the tokio
/// `JoinHandle`), but the continuation carrying its refusal and its re-list
/// never did — a refused rename indistinguishable from an accepted one.
#[gpui::test]
fn a_refresh_landing_mid_rename_keeps_the_spaces_ops_error(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let real = core
        .runtime()
        .block_on(core.create_space(Some("Nile".into())))
        .expect("create space")
        .id;

    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(None, Ok(vec![stale_row("ghost")]), None, cx)
    });
    stores
        .spaces
        .update(cx, |s, cx| s.rename("ghost".into(), "Renamed".into(), cx));
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::SpaceIndex), cx));

    wait_until(cx, "the refusal surfaces and the index reconciles", |cx| {
        stores.spaces.read_with(cx, |s, _| {
            s.op_error().is_some() && s.list().iter().any(|r| r.id == real)
        })
    });
    stores.spaces.read_with(cx, |s, _| {
        assert!(
            s.list().iter().any(|r| r.id == real),
            "the mutation's batch-end read landed on its failure exit"
        );
        assert!(
            !s.index().is_loading(),
            "the mutation must resolve the cell it took the read from"
        );
    });
}

/// **Two mutations on two spaces are independent work.** A single mutation slot
/// made the second supersede the first — dropping the gpui half of a write
/// already in flight (its re-list and its refusal lost: the silent-loss class
/// this store's write-through shape exists to cure), or cancelling the first
/// operation outright when it had not been polled yet. Rapid ops from two
/// Library rows or two windows are the realistic trigger; issued in one update
/// here, which is the narrowest form of it (the first has not run at all).
#[gpui::test]
fn two_renames_on_two_spaces_both_land(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let a = core
        .runtime()
        .block_on(core.create_space(Some("A".into())))
        .expect("create space")
        .id;
    let b = core
        .runtime()
        .block_on(core.create_space(Some("B".into())))
        .expect("create space")
        .id;

    stores.spaces.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the index loads", |cx| {
        stores.spaces.read_with(cx, |s, _| s.list().len() == 2)
    });

    stores.spaces.update(cx, |s, cx| {
        s.rename(a.clone(), "Anemone".into(), cx);
        s.rename(b.clone(), "Bergamot".into(), cx);
    });

    let durable_titles = |cx: &mut TestAppContext| {
        let _ = cx;
        let listing = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("list spaces");
        let mut titles: Vec<String> = listing.iter().filter_map(|s| s.title.clone()).collect();
        titles.sort();
        titles
    };
    wait_until(cx, "both renames land durably", |cx| {
        durable_titles(cx) == vec!["Anemone".to_string(), "Bergamot".to_string()]
    });
    stores.spaces.read_with(cx, |s, _| {
        let title = |id: &str| {
            s.list()
                .iter()
                .find(|r| r.id == id)
                .and_then(|r| r.title.clone())
        };
        assert_eq!(title(&a).as_deref(), Some("Anemone"));
        assert_eq!(title(&b).as_deref(), Some("Bergamot"));
    });
}

/// The refusal half of the same rule: two refused mutations on two spaces each
/// owe their own report. One tagged slot could only hold the last, and the
/// *inspector* reads per space — a lost refusal there is a title field snapping
/// back with no reason given, exactly the dishonesty this store's shape exists
/// to prevent. A bus refresh lands mid-write for good measure: it must defer to
/// the **last** mutation, not eat either one's continuation.
#[gpui::test]
fn two_refused_mutations_each_keep_their_own_refusal(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let real = core
        .runtime()
        .block_on(core.create_space(Some("Nile".into())))
        .expect("create space")
        .id;

    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(
            None,
            Ok(vec![stale_row("ghost-a"), stale_row("ghost-b")]),
            None,
            cx,
        )
    });
    stores.spaces.update(cx, |s, cx| {
        s.rename("ghost-a".into(), "Anemone".into(), cx);
        s.rename("ghost-b".into(), "Bergamot".into(), cx);
    });
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::SpaceIndex), cx));

    wait_until(cx, "both refusals surface and the index reconciles", |cx| {
        stores.spaces.read_with(cx, |s, _| {
            s.op_error_for("ghost-a").is_some()
                && s.op_error_for("ghost-b").is_some()
                && s.list().len() == 1
        })
    });
    stores.spaces.read_with(cx, |s, _| {
        assert!(
            s.op_error().is_some(),
            "the Library shows one of them (the most recent)"
        );
        assert!(
            s.list().iter().any(|r| r.id == real) && s.list().len() == 1,
            "and the batch-end read landed: the index is the database's"
        );
        assert!(!s.index().is_loading());
    });
}

/// **A superseded same-space write leaves no optimism behind.** Two mutations on
/// one space still replace-cancel (one control, last-wins — the documented
/// residual), and the loser's undo dies with its cancelled task. Its *edit* must
/// not outlive it: the successor would otherwise treat the loser's optimistic
/// string as the value to restore, and a refused successor whose re-list also
/// failed would put back a name the database never held. Here the winner is
/// refused (a stale id), so the reconciled index must show the database's title
/// — never either optimistic one.
#[gpui::test]
fn a_superseded_rename_leaves_no_optimism_behind(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let real = core
        .runtime()
        .block_on(core.create_space(Some("Tides".into())))
        .expect("create space")
        .id;

    // The cached index holds a row the database does not, so both writes below
    // are refused (`rename_space` on a stale id).
    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(None, Ok(vec![stale_row("ghost")]), None, cx)
    });
    stores.spaces.update(cx, |s, cx| {
        s.rename("ghost".into(), "Anemone".into(), cx);
        s.rename("ghost".into(), "Bergamot".into(), cx);
    });

    wait_until(cx, "the refusal surfaces and the index reconciles", |cx| {
        stores.spaces.read_with(cx, |s, _| {
            s.op_error_for("ghost").is_some() && !s.list().iter().any(|r| r.id == "ghost")
        })
    });
    stores.spaces.read_with(cx, |s, _| {
        let titles: Vec<String> = s.list().iter().filter_map(|r| r.title.clone()).collect();
        assert_eq!(
            titles,
            vec!["Tides".to_string()],
            "the index is the database's listing — neither optimistic name survives"
        );
        assert!(s.list().iter().any(|r| r.id == real));
        assert!(!s.index().is_loading());
    });
}

/// **The index a batch lands on is read after the batch's last write.** Each
/// mutation used to carry a listing it read right after *its own* write; with
/// siblings in flight, the operation still standing when the last slot cleared
/// could be one whose read predated another's commit, and it resolved the shared
/// index with that snapshot — dropping an accepted rename from view (and, when a
/// read failed, preserving the stale snapshot as `Failed { prior }`). The
/// resolving read is now issued only once no write is in flight, so "after every
/// write of the batch" is a property of when it is taken.
///
/// **The assertion has to be a stability one.** A batch's optimistic edits agree
/// with the database the moment its writes commit, so comparing once passes long
/// before the resolution lands — the defect is precisely a *later* read putting
/// something staler on screen. So the comparison is held for a while after it
/// first holds, and any drift fails.
#[gpui::test]
fn a_batch_of_renames_lands_on_a_listing_read_after_all_of_them(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let a = core
        .runtime()
        .block_on(core.create_space(Some("A".into())))
        .expect("create space")
        .id;
    let b = core
        .runtime()
        .block_on(core.create_space(Some("B".into())))
        .expect("create space")
        .id;

    stores.spaces.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the index loads", |cx| {
        stores.spaces.read_with(cx, |s, _| s.list().len() == 2)
    });

    for round in 0..4 {
        let title_a = format!("A{round}");
        let title_b = format!("B{round}");
        let (ta, tb) = (title_a.clone(), title_b.clone());
        stores.spaces.update(cx, |s, cx| {
            s.rename(a.clone(), ta, cx);
            s.rename(b.clone(), tb, cx);
        });

        // The store's index beside the database's own listing.
        let agrees = |cx: &mut TestAppContext| -> Result<(), String> {
            let mut durable: Vec<(String, Option<String>)> = core
                .runtime()
                .block_on(core.list_spaces(false))
                .expect("list spaces")
                .into_iter()
                .map(|s| (s.id, s.title))
                .collect();
            durable.sort();
            let mut cached = stores.spaces.read_with(cx, |s, _| {
                s.list()
                    .iter()
                    .map(|r| (r.id.clone(), r.title.clone()))
                    .collect::<Vec<_>>()
            });
            cached.sort();
            if cached != durable {
                return Err(format!("cached {cached:?} != durable {durable:?}"));
            }
            Ok(())
        };

        wait_until(cx, "the batch reaches the database's listing", |cx| {
            agrees(cx).is_ok()
        });
        // …and stays there. A resolution taken from a snapshot older than the
        // batch's last write lands after this point and would drift back.
        for _ in 0..40 {
            cx.run_until_parked();
            std::thread::sleep(std::time::Duration::from_millis(25));
            if let Err(drift) = agrees(cx) {
                panic!("the index drifted after the batch settled: {drift}");
            }
        }
    }
}

/// `Change::Participants` fans out to **three** stores, and the agent library
/// is the third (task 36): a promotion or retirement changes what the library
/// lists, and an "edit everywhere" changes what a row says. Neither store polls;
/// the dispatcher is what makes one signal answer for two domains.
#[gpui::test]
fn a_participants_change_reaches_the_agent_library(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    stores
        .agents
        .read_with(cx, |a, _| assert!(!a.agents().is_loading()));

    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx));
    stores.agents.read_with(cx, |a, _| {
        assert!(
            a.agents().is_loading(),
            "a Change::Participants must drive AgentsStore::refresh"
        );
    });

    // A templates change is not one of its signals — the library holds no
    // template rows.
    let (other, _dir2) = backed_stores(cx);
    cx.update(|cx| stores::dispatch_change_for_test(&other, Some(Change::Templates), cx));
    other.agents.read_with(cx, |a, _| {
        assert!(
            !a.agents().is_loading(),
            "a Change::Templates must not touch the agent library"
        );
    });
}

/// Promote each space's seeded agent and hand back the two participant ids —
/// the only way an agent reaches the library.
fn two_shared_agents(cx: &mut TestAppContext, stores: &Stores) -> (String, String) {
    let core = stores.app_core().expect("backed stores carry a core");
    let mut ids = Vec::new();
    for title in ["A", "B"] {
        let space = core
            .runtime()
            .block_on(core.create_space(Some(title.into())))
            .expect("create space")
            .id;
        let agent = core
            .runtime()
            .block_on(core.list_space_participants(space))
            .expect("participants")
            .into_iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .id;
        core.runtime()
            .block_on(core.promote_participant(agent.clone(), None, None))
            .expect("promotion");
        ids.push(agent);
    }
    stores.agents.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the library lists both", |cx| {
        stores.agents.read_with(cx, |s, _| s.list().len() == 2)
    });
    (ids[0].clone(), ids[1].clone())
}

/// **Two writes on two agents both land** (Codex review, PR #279). The Agents
/// pane offers Edit and Retire on every row at once, so editing one agent and
/// retiring another are independent operations a reader can start a moment
/// apart. With one store-wide slot the second replaced the first: unpolled, the
/// first write simply never happened; polled, it ran core-side while its
/// refusal and its re-list were discarded. Keyed slots are what make them
/// independent — and the resolving read is taken once the **last** one settles.
#[gpui::test]
fn two_writes_on_two_agents_both_land(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let (a, b) = two_shared_agents(cx, &stores);

    // Both issued before either future is polled — the narrowest interleaving.
    stores.agents.update(cx, |s, cx| {
        s.update_agent(
            a.clone(),
            eidola_app_core::ParticipantUpdate {
                label: Some("Ada".into()),
                ..Default::default()
            },
            cx,
        );
        s.retire(b.clone(), cx);
    });

    let durable = |_cx: &mut TestAppContext| {
        core.runtime()
            .block_on(core.list_global_agents())
            .expect("library")
    };
    wait_until(cx, "both writes land durably", |cx| {
        let rows = durable(cx);
        rows.len() == 1 && rows[0].label == "Ada"
    });
    // And the roster the pane reads agrees — the batch-end read was taken after
    // the last write, so it carries both.
    wait_until(cx, "the roster resolves on a read after both", |cx| {
        stores
            .agents
            .read_with(cx, |s, _| s.list().len() == 1 && s.list()[0].label == "Ada")
    });
    stores.agents.read_with(cx, |s, _| {
        assert!(s.op_error(&a).is_none(), "the edit was accepted");
        assert!(s.op_error(&b).is_none(), "the retirement was accepted");
    });
}

/// The refusal half: two refused writes each keep their own report. One
/// store-wide slot could hold only the last, and the pane renders a band **per
/// row** — a lost refusal there is an edit that silently did nothing under the
/// row the reader was looking at.
#[gpui::test]
fn two_refused_agent_writes_each_keep_their_own_refusal(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let (a, b) = two_shared_agents(cx, &stores);

    // An invalid notify policy is refused before any write (zero trace).
    let refused = |label: &str| eidola_app_core::ParticipantUpdate {
        label: Some(label.into()),
        notify_policy: Some("sometimes".into()),
        ..Default::default()
    };
    stores.agents.update(cx, |s, cx| {
        s.update_agent(a.clone(), refused("Ada"), cx);
        s.update_agent(b.clone(), refused("Bo"), cx);
    });

    wait_until(cx, "both refusals stand", |cx| {
        stores.agents.read_with(cx, |s, _| {
            s.op_error(&a).is_some() && s.op_error(&b).is_some()
        })
    });
    // Each × acknowledges its own, and leaves the other standing.
    stores.agents.update(cx, |s, cx| s.clear_op_error(&a, cx));
    stores.agents.read_with(cx, |s, _| {
        assert!(s.op_error(&a).is_none());
        assert!(
            s.op_error(&b).is_some(),
            "dismissing one refusal must not discard the other"
        );
    });
    // Neither refusal wrote anything: both agents are still there, unrenamed.
    stores.agents.read_with(cx, |s, _| {
        assert_eq!(s.list().len(), 2);
        assert!(!s.list().iter().any(|x| x.label == "Ada" || x.label == "Bo"));
    });
}

/// **One `bridge` closure is not one transaction** (Codex review, PR #279).
///
/// The share's two core calls travel together so no gpui-side refresh can land
/// between them — but their *database* writes are still two, and two windows
/// share one `AppCore`. Let another window promote (or a Settings retire remove)
/// the same agent after the editor took its snapshot, and the persona write
/// lands on a row the promotion then refuses: the reader is told sharing failed
/// while their draft was persisted — across **every** space, since the row they
/// wrote to is now a global one. The cure is that the persona travels *into*
/// `AppCore::promote_participant`, which applies it inside the promoting
/// transaction, behind the same `scope = 'space'` guard — so a lost race rolls
/// the persona back with it and the refusal leaves zero trace.
#[gpui::test]
fn a_share_that_loses_the_race_writes_no_persona(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(Some("A".into())))
        .expect("create space")
        .id;
    let agent = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the default template seeds one agent")
        .id;

    // The other window shares it first. The editor open here still believes it
    // is looking at a space-owned agent — that snapshot is what makes the draft
    // stale, and nothing about the two-call shape can notice.
    core.runtime()
        .block_on(core.promote_participant(agent.clone(), None, None))
        .expect("the other window's share");
    let before = core
        .runtime()
        .block_on(core.list_global_agents())
        .expect("library")
        .into_iter()
        .find(|a| a.id == agent)
        .expect("the shared agent");

    stores.participants.update(cx, |s, cx| {
        s.promote(
            space.clone(),
            agent.clone(),
            Some(eidola_app_core::ParticipantUpdate {
                label: Some("Cartographer".into()),
                system_prompt: Some(Some("Draw the map before arguing about it.".into())),
                ..Default::default()
            }),
            cx,
        )
    });
    wait_until(cx, "the share is refused", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.op_errors_for(&space).is_empty())
    });

    let after = core
        .runtime()
        .block_on(core.list_global_agents())
        .expect("library")
        .into_iter()
        .find(|a| a.id == agent)
        .expect("the shared agent is still there");
    assert_eq!(
        after.label, before.label,
        "a refused share must not rename the agent it failed to share"
    );
    assert_eq!(
        after.system_prompt, before.system_prompt,
        "nor rewrite its charter — across every space that follows the shared row"
    );
}

/// **A grant decides at the write, not from the picker's snapshot** (Codex
/// review, PR #280). The invite form records whether a candidate was shared
/// when its listing landed; another window sharing it before the reader
/// confirms made that snapshot a lie, and a store that branched on it asked for
/// a promotion of an already-global row — refused, for a membership app-core
/// would have added without complaint. The store now names the *outcome* it
/// wants and lets one transaction decide the verb.
#[gpui::test]
fn a_grant_that_loses_the_race_to_a_promotion_still_adds_the_membership(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let home = core
        .runtime()
        .block_on(core.create_space(Some("Home".into())))
        .expect("create space")
        .id;
    let here = core
        .runtime()
        .block_on(core.create_space(Some("Here".into())))
        .expect("create space")
        .id;
    let agent = core
        .runtime()
        .block_on(core.list_space_participants(home.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the default template seeds one agent")
        .id;

    // The form listed it as space-owned; another window shares it meanwhile.
    core.runtime()
        .block_on(core.promote_participant(agent.clone(), None, None))
        .expect("the other window's share");

    stores.participants.update(cx, |s, cx| {
        s.grant_membership(here.clone(), agent.clone(), cx)
    });
    wait_until(cx, "the grant lands", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| s.list(&here).iter().any(|p| p.id == agent))
    });
    assert!(
        stores
            .participants
            .read_with(cx, |s, _| s.op_errors_for(&here).is_empty()),
        "a grant onto a row someone else shared first is not a refusal"
    );
}

/// **Two writes on two participants of one space both land** (Codex review, PR
/// #279). The inspector offers a verb on every roster row at once — share this
/// agent, remove that one — so two mutations a moment apart are independent.
/// `ParticipantsStore` keyed its mutation slot by **space** alone, so the second
/// replaced the first: unpolled, the first write simply never ran; polled, its
/// refusal and its re-list were discarded. Keyed by `(space, participant)` they
/// are independent, and the resolving read is taken once that space's last write
/// settles.
#[gpui::test]
fn two_writes_on_two_participants_of_one_space_both_land(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(Some("A".into())))
        .expect("create space")
        .id;
    let seeded = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the default template seeds one agent")
        .id;
    let second = core
        .runtime()
        .block_on(core.add_space_participant(
            space.clone(),
            eidola_app_core::NewParticipant {
                label: "Bo".into(),
                model_ref: None,
                system_prompt: None,
                notify_policy: "explicit".into(),
            },
        ))
        .expect("a second agent")
        .id;

    // Both issued before either future is polled — the narrowest interleaving.
    stores.participants.update(cx, |s, cx| {
        s.promote(space.clone(), seeded.clone(), None, cx);
        s.remove(space.clone(), second.clone(), cx);
    });

    wait_until(cx, "both writes land durably", |_cx| {
        let rows = core
            .runtime()
            .block_on(core.list_space_participants(space.clone()))
            .expect("participants");
        rows.iter()
            .any(|p| p.id == seeded && p.source == "referenced")
            && !rows.iter().any(|p| p.id == second)
    });
    // And the roster the panel reads agrees — the batch-end read was taken after
    // the last of them, so it carries both.
    wait_until(cx, "the roster resolves on a read after both", |cx| {
        stores.participants.read_with(cx, |s, _| {
            let rows = s.list(&space);
            rows.iter()
                .any(|p| p.id == seeded && p.source == "referenced")
                && !rows.iter().any(|p| p.id == second)
        })
    });
    assert!(
        stores
            .participants
            .read_with(cx, |s, _| s.op_errors_for(&space).is_empty()),
        "both writes were accepted"
    );
}

/// The refusal half: two refused writes in one space each keep their own report.
/// The inspector renders **one band per space** by design (the panel is 320px),
/// so the store keeps every standing refusal and the band lists them, each named
/// — where a single slot could only have shown the last, and the reader would
/// never learn their other action was refused too.
#[gpui::test]
fn two_refused_participant_writes_each_keep_their_own_refusal(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(Some("A".into())))
        .expect("create space")
        .id;
    let rows = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants");
    let agent = rows
        .iter()
        .find(|p| p.kind == "agent")
        .expect("the seeded agent")
        .id
        .clone();
    let human = rows
        .iter()
        .find(|p| p.kind == "human")
        .expect("the shared You")
        .id
        .clone();

    // Two refusals app-core decides for itself: a blank label, and removing the
    // shared human.
    stores.participants.update(cx, |s, cx| {
        s.update_everywhere(
            space.clone(),
            agent.clone(),
            eidola_app_core::ParticipantUpdate {
                label: Some("   ".into()),
                ..Default::default()
            },
            eidola_app_core::ExpectedScope::Any,
            cx,
        );
        s.remove(space.clone(), human.clone(), cx);
    });

    wait_until(cx, "both refusals stand", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| s.op_errors_for(&space).len() == 2)
    });
    stores.participants.read_with(cx, |s, _| {
        let subjects: Vec<String> = s
            .op_errors_for(&space)
            .into_iter()
            .map(|(pid, _)| pid)
            .collect();
        assert!(
            subjects.contains(&agent),
            "the edit's refusal: {subjects:?}"
        );
        assert!(
            subjects.contains(&human),
            "the removal's refusal: {subjects:?}"
        );
    });
}

/// **A Save carries the premise it was composed under** (Codex review, PR #279).
///
/// Save and Share are the same control on one row, so the second replaces the
/// first's slot — sanctioned last-wins. But a replaced write's *core* call keeps
/// running, and the two writes do not share a premise: the Save was composed
/// against a **space-owned** row, and if the promotion commits first the stale
/// Save lands on a row that is now **global**, republishing the old persona to
/// every space the agent joins. Liveness alone cannot see that; the premise has
/// to ride the write.
#[gpui::test]
fn a_stale_owned_save_refuses_once_the_row_is_shared(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space = core
        .runtime()
        .block_on(core.create_space(Some("A".into())))
        .expect("create space")
        .id;
    let agent = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the seeded agent")
        .id;
    let before = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.id == agent)
        .expect("the agent")
        .label;

    // The other window's Share lands first.
    core.runtime()
        .block_on(core.promote_participant(agent.clone(), None, None))
        .expect("the share");

    // The Save that was already in flight, composed against the owned row.
    stores.participants.update(cx, |s, cx| {
        s.update_everywhere(
            space.clone(),
            agent.clone(),
            eidola_app_core::ParticipantUpdate {
                label: Some("Stale name from the owned editor".into()),
                ..Default::default()
            },
            eidola_app_core::ExpectedScope::SpaceOwned {
                space_id: space.clone(),
            },
            cx,
        )
    });
    wait_until(cx, "the stale save is refused", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.op_errors_for(&space).is_empty())
    });
    let after = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.id == agent)
        .expect("the agent");
    assert_eq!(
        after.label, before,
        "a save composed against an owned row must not rewrite the shared identity"
    );
    assert_eq!(after.scope, "global");
}

/// **Two writes on one control are sequenced, not raced** (Codex review, PR
/// #279). Replace-cancel was documented as "last-wins", but it only dropped the
/// *gpui* half: `bridge` leaves the core write running, so the superseded write
/// could reach the database after its successor — last-wins reversed — or, when
/// the successor arrived before the first was ever polled, vanish entirely.
///
/// The successor now takes ownership of its predecessor's task and awaits it, so
/// it starts strictly after that write's round trip. Both land, in order: the
/// first's charter survives (it ran), and the second's name is what stands (it
/// ran last).
#[gpui::test]
fn two_saves_on_one_agent_are_sequenced(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let (agent, _b) = two_shared_agents(cx, &stores);

    stores.agents.update(cx, |s, cx| {
        s.update_agent(
            agent.clone(),
            eidola_app_core::ParticipantUpdate {
                label: Some("First".into()),
                system_prompt: Some(Some("Written by the first save.".into())),
                ..Default::default()
            },
            cx,
        );
        s.update_agent(
            agent.clone(),
            eidola_app_core::ParticipantUpdate {
                label: Some("Second".into()),
                ..Default::default()
            },
            cx,
        );
    });

    wait_until(cx, "both saves land in order", |_cx| {
        core.runtime()
            .block_on(core.list_global_agents())
            .expect("library")
            .iter()
            .any(|a| a.id == agent && a.label == "Second")
    });
    let row = core
        .runtime()
        .block_on(core.list_global_agents())
        .expect("library")
        .into_iter()
        .find(|a| a.id == agent)
        .expect("the agent");
    assert_eq!(
        row.system_prompt.as_deref(),
        Some("Written by the first save."),
        "the superseded save still ran — its charter is the proof"
    );
    assert_eq!(row.label, "Second", "and the later save is what stands");
}

/// REGRESSION (Codex review, PR #292): **a transcript is where an author's name
/// lives.** `get_space_tree` resolves every post's byline — and every reference
/// edge's carried author identity — through
/// `COALESCE(space_participant.override_label, participant.label)` at read
/// time, and nothing re-derives them afterwards. A rename emits
/// `Change::Participants` and nothing else, so before this every open window
/// went on showing the old name in its post gutters, its minimap labels and its
/// footnote rails until an unrelated `Change::Space` or a reopen.
///
/// The signal carries no id (a global agent is renamed everywhere at once, and
/// an override is written per space), so the answer is **every** live space.
#[gpui::test]
fn a_participants_change_re_reads_a_live_transcript(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let posted = core
        .runtime()
        .block_on(core.post("The tide is the moon's doing.".into(), None))
        .expect("post");
    let space = stores
        .spaces
        .update(cx, |s, cx| s.open(posted.space_id.clone(), cx));

    let byline = |cx: &mut TestAppContext| {
        space.read_with(cx, |s, _| s.messages().first().map(|m| m.byline.clone()))
    };
    wait_until(cx, "the transcript loads", |cx| byline(cx).is_some());
    assert_eq!(
        byline(cx).as_deref(),
        Some("You"),
        "the seeded human's own name for itself"
    );

    // Another window (or the CLI) renames that participant *in this space*.
    core.runtime()
        .block_on(core.set_space_participant_override(
            posted.space_id.clone(),
            eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            eidola_app_core::ParticipantOverride {
                label: Some(Some("Skipper".into())),
                model_ref: None,
                system_prompt: None,
                notify_policy: None,
            },
        ))
        .expect("override");

    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx));
    wait_until(cx, "the transcript re-reads", |cx| {
        byline(cx).as_deref() == Some("Skipper")
    });
}

/// REGRESSION (Codex review, PR #292, round 2): the **lagged** recovery reaches
/// the live conversations, not just the Library index.
///
/// `refresh_everything` is the doctrine's answer to a dropped change — we no
/// longer know what we missed, so re-read everything. But `SpacesStore::refresh`
/// re-reads the *index*; the transcripts belong to the registered `Space`
/// entities, and nothing told them anything. So a lag that swallowed a rename
/// left every already-open window naming its authors as it did when it loaded —
/// post gutters, minimap labels and footnote rails alike — indefinitely, while
/// the recovery claimed to have refreshed all state.
#[gpui::test]
fn a_lagged_bus_re_reads_a_live_transcript(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let posted = core
        .runtime()
        .block_on(core.post("The tide is the moon's doing.".into(), None))
        .expect("post");
    let space = stores
        .spaces
        .update(cx, |s, cx| s.open(posted.space_id.clone(), cx));

    let byline = |cx: &mut TestAppContext| {
        space.read_with(cx, |s, _| s.messages().first().map(|m| m.byline.clone()))
    };
    wait_until(cx, "the transcript loads", |cx| byline(cx).is_some());
    assert_eq!(byline(cx).as_deref(), Some("You"));

    core.runtime()
        .block_on(core.set_space_participant_override(
            posted.space_id.clone(),
            eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            eidola_app_core::ParticipantOverride {
                label: Some(Some("Skipper".into())),
                model_ref: None,
                system_prompt: None,
                notify_policy: None,
            },
        ))
        .expect("override");

    // The bus dropped it: no `Change` names what changed, only that something
    // did (`dispatch_change_for_test(None)` is `Lagged`).
    cx.update(|cx| stores::dispatch_change_for_test(&stores, None, cx));
    wait_until(cx, "the lagged recovery re-reads the transcript", |cx| {
        byline(cx).as_deref() == Some("Skipper")
    });
}

/// REGRESSION (Codex review, PR #292, round 2): a participant change arriving
/// while a space is posting or streaming is **deferred, not discarded**.
///
/// The operation's own exit reload is not synchronization: `get_space_tree` is
/// several reads, so a rename committing while they run is captured by none of
/// them — and the event announcing it was dropped on the busy gate, leaving
/// nothing to correct the stale names afterwards. The invalidation is now
/// recorded and replayed once the last mutation or turn settles.
#[gpui::test]
fn a_participants_change_arriving_mid_operation_is_replayed(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let posted = core
        .runtime()
        .block_on(core.post("The tide is the moon's doing.".into(), None))
        .expect("post");
    let space = stores
        .spaces
        .update(cx, |s, cx| s.open(posted.space_id.clone(), cx));

    let byline = |cx: &mut TestAppContext| {
        space.read_with(cx, |s, _| s.messages().first().map(|m| m.byline.clone()))
    };
    wait_until(cx, "the transcript loads", |cx| byline(cx).is_some());
    assert_eq!(byline(cx).as_deref(), Some("You"));

    // A mutation owns the transcript's truth.
    space.update(cx, |s, cx| s.arm_post_runner_for_test(cx));

    core.runtime()
        .block_on(core.set_space_participant_override(
            posted.space_id.clone(),
            eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            eidola_app_core::ParticipantOverride {
                label: Some(Some("Skipper".into())),
                model_ref: None,
                system_prompt: None,
                notify_policy: None,
            },
        ))
        .expect("override");
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx));
    cx.run_until_parked();
    assert_eq!(
        byline(cx).as_deref(),
        Some("You"),
        "the busy gate still holds the reload off while the operation runs"
    );

    // The operation settles — and the debt it deferred is discharged.
    space.update(cx, |s, cx| s.clear_post_runner_for_test(cx));
    wait_until(cx, "the deferred invalidation is replayed", |cx| {
        byline(cx).as_deref() == Some("Skipper")
    });
}

/// **A write nobody is waiting on is deferred while the space is busy, and
/// replayed** — the `Change::Space` half of the same rule the participants
/// change above obeys.
///
/// The exit reload an in-flight operation performs is several reads, so a post
/// committing while they run is caught by none of them; dropping its signal
/// loses it until something unrelated re-reads. The one that reaches a busy
/// window in practice is app-core's own sub-space driver posting a delegation's
/// report into the conversation the reader is mid-turn in.
#[gpui::test]
fn an_unattended_space_change_arriving_mid_operation_is_replayed(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let posted = core
        .runtime()
        .block_on(core.post("The tide is the moon's doing.".into(), None))
        .expect("post");
    let space = stores
        .spaces
        .update(cx, |s, cx| s.open(posted.space_id.clone(), cx));

    let count = |cx: &mut TestAppContext| space.read_with(cx, |s, _| s.messages().len());
    wait_until(cx, "the transcript loads", |cx| count(cx) == 1);

    // A turn owns the transcript's truth.
    space.update(cx, |s, cx| s.arm_post_runner_for_test(cx));

    // Something else writes into this very space while that runs.
    core.runtime()
        .block_on(core.post(
            "And the moon is nobody's.".into(),
            Some(posted.space_id.clone()),
        ))
        .expect("a second post");
    cx.update(|cx| {
        stores::dispatch_change_event_for_test(
            &stores,
            Some(Change::Space(posted.space_id.clone())),
            ChangeOrigin::Unattended,
            cx,
        )
    });
    cx.run_until_parked();
    assert_eq!(
        count(cx),
        1,
        "the busy gate still holds the reload off while the operation runs"
    );

    space.update(cx, |s, cx| s.clear_post_runner_for_test(cx));
    wait_until(cx, "the deferred invalidation is replayed", |cx| {
        count(cx) == 2
    });
}

/// **And an own write buys no redundant re-read.** Every post raises
/// `Change::Space`, and almost all of them are the busy operation's own — its
/// exit reload already covers those, so recording them would cost a whole-tree
/// re-read at the end of every turn to close a window only a write from
/// somewhere else can open.
#[gpui::test]
fn a_caller_space_change_arriving_mid_operation_is_dropped(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let posted = core
        .runtime()
        .block_on(core.post("The tide is the moon's doing.".into(), None))
        .expect("post");
    let space = stores
        .spaces
        .update(cx, |s, cx| s.open(posted.space_id.clone(), cx));

    let count = |cx: &mut TestAppContext| space.read_with(cx, |s, _| s.messages().len());
    wait_until(cx, "the transcript loads", |cx| count(cx) == 1);

    space.update(cx, |s, cx| s.arm_post_runner_for_test(cx));
    core.runtime()
        .block_on(core.post(
            "And the moon is nobody's.".into(),
            Some(posted.space_id.clone()),
        ))
        .expect("a second post");
    cx.update(|cx| {
        stores::dispatch_change_for_test(&stores, Some(Change::Space(posted.space_id.clone())), cx)
    });
    cx.run_until_parked();

    // The operation settles with no debt outstanding: the reload that lands is
    // the operation's own, not one this signal bought.
    space.update(cx, |s, cx| s.clear_post_runner_for_test(cx));
    cx.run_until_parked();
    assert_eq!(
        count(cx),
        1,
        "a caller-origin change is contained by the operation's own exit reload"
    );
}

/// A ⌘N space's row commits behind its window, and the very first Send must
/// not race it.
///
/// The store's insert and the entity's post are independent work on the same
/// runtime, so nothing but an explicit wait orders them — and losing that race
/// is not a delay, it is a loss: `post` refuses with "space not found", the
/// failure completion reloads the transcript, and the thought the reader had
/// already typed disappears from a space that then exists perfectly well.
///
/// The interleaving is staged rather than hoped for: creating and submitting in
/// the same pass means the creation task has not been polled yet, so its insert
/// has not reached the runtime when the post's does.
#[gpui::test]
fn a_first_post_into_a_brand_new_space_waits_out_its_insert(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let space = stores.spaces.update(cx, |s, cx| s.create(cx));
    let accepted = space.update(cx, |s, cx| {
        s.submit("The tide is the moon's doing.".into(), None, Vec::new(), cx)
    });
    assert!(accepted, "the composer's first post is accepted");
    let space_id = space.read_with(cx, |s, _| s.id().to_string());

    wait_until(
        cx,
        "the post lands durably in the space ⌘N created",
        |_| {
            core.runtime()
                .block_on(core.get_space_messages(space_id.clone()))
                .is_ok_and(|m| m.len() == 1)
        },
    );
    space.read_with(cx, |s, _| {
        assert_eq!(
            s.messages().len(),
            1,
            "and the thought the reader typed is still on screen"
        );
    });
}

/// The mirror of the test above: a save into a space whose creation was
/// **refused** fails at once with that refusal, rather than waiting forever on
/// a settle that has already happened — or writing into whatever the id turns
/// out to name.
#[gpui::test]
fn a_post_into_a_space_that_was_never_created_is_refused(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let space = stores.spaces.update(cx, |s, cx| s.create(cx));
    let space_id = space.read_with(cx, |s, _| s.id().to_string());
    stores.spaces.update(cx, |s, cx| {
        s.fail_creation_for_test(
            &space_id,
            eidola_app_core::error::AppError::Database {
                message: "disk is full".into(),
            },
            cx,
        )
    });

    // The insert this test countermands still lands (nothing cancels a `bridge`
    // call), so the id really does name a row — which is what makes "the post
    // was not written" a statement about the refusal rather than about a
    // missing space.
    wait_until(cx, "the underlying row commits", |_| {
        core.runtime()
            .block_on(core.list_spaces(false))
            .is_ok_and(|spaces| spaces.iter().any(|s| s.id == space_id))
    });

    assert!(
        space.update(cx, |s, cx| s.submit(
            "into thin air".into(),
            None,
            Vec::new(),
            cx
        )),
        "the draft is still accepted — the refusal is reported, not guessed at"
    );
    for _ in 0..40 {
        cx.run_until_parked();
        if !space.read_with(cx, |s, _| s.is_busy()) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    assert!(
        !space.read_with(cx, |s, _| s.is_busy()),
        "the save settles instead of waiting on a creation that already failed"
    );
    assert!(
        core.runtime()
            .block_on(core.get_space_messages(space_id))
            .is_ok_and(|m| m.is_empty()),
        "and nothing was written"
    );
}

/// REGRESSION (Codex review, PR #292, round 3): the deferral above has to reach
/// a space that was created a moment ago and whose first save is in flight.
///
/// That window is exactly when a brand-new space's transcript is read — the
/// save's own `get_space_tree` is several reads, so a rename committing while
/// they run is captured by none of them — and the `Change::Participants`
/// announcing it arrives while the space is still busy. The delivery half is
/// now structural: `SpacesStore::create` registers the entity under its id in
/// the same breath as it mints it, so a store-wide broadcast reaches it by
/// ordinary registry membership rather than by a second collection anyone
/// could forget to sweep. The keeping half is still the entity's: the busy
/// gate defers rather than discards, and the save's exit replays it.
#[gpui::test]
fn a_participants_change_reaches_a_brand_new_space_mid_save(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    // ⌘N: the id is minted here and the row commits behind the window.
    let space = stores.spaces.update(cx, |s, cx| s.create(cx));
    let space_id = space.read_with(cx, |s, _| s.id().to_string());
    // The registry holds it from this frame — before the insert has landed.
    let rejoined = stores
        .spaces
        .update(cx, |s, cx| s.open(space_id.clone(), cx));
    assert_eq!(
        rejoined.entity_id(),
        space.entity_id(),
        "a new space is an ordinary registry member from the first frame"
    );
    wait_until(cx, "the space's row commits", |_| {
        core.runtime()
            .block_on(core.list_spaces(false))
            .is_ok_and(|spaces| spaces.iter().any(|s| s.id == space_id))
    });

    // Its first post, which the entity's save is about to reload around.
    core.runtime()
        .block_on(core.post(
            "The tide is the moon's doing.".into(),
            Some(space_id.clone()),
        ))
        .expect("post");

    // Stage what the save's own multi-read tree query captured — the tree as it
    // read *before* the rename below commits.
    let captured = core
        .runtime()
        .block_on(core.get_space_tree(space_id.clone()))
        .expect("tree");
    space.update(cx, |s, cx| {
        s.set_post_tree_for_test(captured, cx);
        // Its first save owns the transcript's truth for the rest of the window.
        s.arm_post_runner_for_test(cx);
    });
    let byline = |cx: &mut TestAppContext| {
        space.read_with(cx, |s, _| s.messages().first().map(|m| m.byline.clone()))
    };
    assert_eq!(
        byline(cx).as_deref(),
        Some("You"),
        "the names the in-flight read captured"
    );

    // The rename commits while the save runs.
    core.runtime()
        .block_on(core.set_space_participant_override(
            space_id.clone(),
            eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            eidola_app_core::ParticipantOverride {
                label: Some(Some("Skipper".into())),
                model_ref: None,
                system_prompt: None,
                notify_policy: None,
            },
        ))
        .expect("override");
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx));
    cx.run_until_parked();
    assert_eq!(
        byline(cx).as_deref(),
        Some("You"),
        "the busy gate still holds the reload off while the save runs"
    );

    // The save lands, and the debt it deferred is discharged with it.
    space.update(cx, |s, cx| s.finish_save_for_test(cx));
    wait_until(cx, "the deferred invalidation is replayed", |cx| {
        byline(cx).as_deref() == Some("Skipper")
    });
}

/// An identity switch must not leave the previous account's billing standing
/// on screen — not even briefly, and not as a "stale" value.
///
/// Nothing on the bus can invalidate the subscription cell (it is a live
/// server read that persists nothing), and the Account pane may already be
/// open on the previous account's answer when the switch happens. A bare
/// refresh is not enough: `to_loading()` on a `Loaded` cell keeps the value
/// visible as `Loaded { stale }`, which here is the wrong account's answer
/// wearing the right account's name.
#[gpui::test]
fn an_identity_switch_drops_the_previous_accounts_subscription(cx: &mut TestAppContext) {
    use eidola_app_core::{SubscriptionInfo, SubscriptionState};
    use eidola_gui::loadable::Loadable;

    let (stores, _dir) = backed_stores(cx);

    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            Loadable::loaded(SubscriptionInfo {
                state: SubscriptionState::Active,
                status: Some("active".into()),
                current_period_end: None,
            }),
            cx,
        );
    });
    stores.account.read_with(cx, |s, _| {
        assert!(s.subscription().value().is_some(), "fixture is in place");
    });

    stores
        .account
        .update(cx, |s, cx| s.account_identity_changed(cx));

    stores.account.read_with(cx, |s, _| {
        assert!(
            s.subscription().value().is_none(),
            "the previous account's standing must be gone, not merely stale"
        );
        assert!(
            s.subscription().is_loading(),
            "and the new account's standing must already be on its way"
        );
    });
}

/// An account operation that *refuses* must leave the cell alone.
///
/// Creating and linking both reject outright when credentials already exist,
/// so a linked reader who opens Get Started and picks "new account" gets a
/// refusal — and clearing when the request went out blanked an already-open
/// Account pane's billing section for an operation that never happened, with
/// no error branch to put it back.
#[gpui::test]
fn a_refused_account_operation_leaves_the_subscription_alone(cx: &mut TestAppContext) {
    use eidola_app_core::{SubscriptionInfo, SubscriptionState};
    use eidola_gui::loadable::Loadable;

    let (stores, _dir) = backed_stores(cx);

    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            Loadable::loaded(SubscriptionInfo {
                state: SubscriptionState::Active,
                status: Some("active".into()),
                current_period_end: None,
            }),
            cx,
        );
    });

    // Both doors an onboarding reader can take. Neither may touch the cell:
    // whether they succeed is decided in app-core, and only the success path
    // is an identity change.
    let _create = stores
        .account
        .read_with(cx, |s, _| s.request_account_create());
    let _verify = stores.account.read_with(cx, |s, _| {
        s.request_verify_account("acct".into(), "secret".into())
    });

    stores.account.read_with(cx, |s, _| {
        assert!(
            matches!(s.subscription(), Loadable::Loaded { .. }),
            "a request that may be refused must not blank the standing"
        );
        assert_eq!(
            s.subscription().value().map(|v| v.state),
            Some(SubscriptionState::Active),
            "and it must be the same standing, untouched"
        );
    });
}

// ---------------------------------------------------------------------------
// Disposing of untouched spaces — the store's half
// ---------------------------------------------------------------------------

/// Whether the core still holds `space`, archived or not.
fn space_row_exists(core: &std::sync::Arc<AppCore>, space: &str) -> bool {
    core.runtime()
        .block_on(core.list_spaces(true))
        .is_ok_and(|spaces| spaces.iter().any(|s| s.id == space))
}

/// **A window can close over its own insert, and the space is still disposed
/// of.** The store owns the insert precisely so a ⌘N closed a keystroke later
/// leaves a *whole* space rather than half of one — which means the close can
/// arrive while the row does not exist yet, where a disposal would answer "no
/// such space" and leave behind the very orphan it exists to prevent. So the id
/// waits for the insert's completion and is disposed of from there.
///
/// The interleaving is staged rather than hoped for: creating and closing in
/// one pass means the creation task has not been polled yet.
///
/// **The absence of the row proves nothing on its own here**, which is the
/// trap this shape invites: the insert is deliberately unpolled, so the id
/// names nothing at the moment the close arrives, and a bare "it is gone"
/// would be satisfied by the state the test starts in. So both ends are
/// observed instead — the deferral, taken while the row does not exist, and
/// app-core's own report that it *deleted* a space under this id, which only a
/// row that existed and was pristine can produce.
#[gpui::test]
fn a_window_closed_over_its_own_insert_still_leaves_nothing_behind(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let entity = stores.spaces.update(cx, |s, cx| s.create(cx));
    let space_id = entity.read_with(cx, |s, _| s.id().to_string());
    // The entity dies with its last window in production; here the test is the
    // only holder, so it has to let go for the release to mean anything.
    drop(entity);
    stores
        .spaces
        .update(cx, |s, cx| s.window_closed(space_id.clone(), false, cx));

    assert!(
        !space_row_exists(&core, &space_id),
        "the premise: the insert has not been polled, so there is no row yet"
    );
    assert!(
        stores
            .spaces
            .read_with(cx, |s, _| s.disposal_deferred_for_test(&space_id)),
        "so the close defers rather than disposing of a space that is not there"
    );

    wait_until(cx, "the insert commits and the space is deleted", |cx| {
        stores
            .spaces
            .read_with(cx, |s, _| s.disposed_for_test().contains(&space_id))
    });
    assert!(
        !space_row_exists(&core, &space_id),
        "and the row it created is gone"
    );
    assert!(
        !stores
            .spaces
            .read_with(cx, |s, _| s.list().iter().any(|r| r.id == space_id)),
        "as is the Library index's residue"
    );
}

/// **A reopen that lands before the verdict is asked for cancels the
/// disposal** — and the window it opened is released to work normally.
///
/// Store calls and every stretch of a spawned task between its `await` points
/// run on gpui's main thread, and `bridge` issues its core call on the first
/// poll of the future being awaited — so the disposal's registry recheck and
/// the issuance of the delete are one uninterrupted segment. A reopen can only
/// land on one side of it. This is the *before* side: the recheck upgrades a
/// live entity, nothing is judged, and the entity — which `open` constructed
/// with its row undecided, because a disposal was in flight for that id — is
/// told the row is there. The submit at the end is what proves the release:
/// a save waits on exactly that gate, so a space left undecided would hang
/// instead of writing.
#[gpui::test]
fn a_reopen_before_the_verdict_cancels_the_disposal(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let entity = stores.spaces.update(cx, |s, cx| s.create(cx));
    let space_id = entity.read_with(cx, |s, _| s.id().to_string());
    drop(entity);
    wait_until(cx, "the space's row commits", |_| {
        space_row_exists(&core, &space_id)
    });

    // The close, and — before the disposal's task gets a turn — a reopen.
    stores
        .spaces
        .update(cx, |s, cx| s.window_closed(space_id.clone(), false, cx));
    let reopened = stores
        .spaces
        .update(cx, |s, cx| s.open(space_id.clone(), cx));
    assert!(
        reopened.read_with(cx, |s, _| s.row_undecided_for_test()),
        "the window opens undecided — a disposal is in flight for this id, so \
         its reads and writes queue behind the verdict rather than racing it"
    );
    assert!(
        !reopened.read_with(cx, |s, _| s.transcript_visible()),
        "and it has answered nothing, so no composer stands over it"
    );

    settle(cx);
    assert!(
        space_row_exists(&core, &space_id),
        "the conversation someone reopened is not an abandoned one"
    );
    assert!(
        stores
            .spaces
            .read_with(cx, |s, _| s.disposed_for_test().is_empty()),
        "and app-core was never asked to delete it"
    );

    // Released, not merely un-deleted: a save waits on the same gate.
    assert!(reopened.update(cx, |s, cx| {
        s.submit("The tide is the moon's doing.".into(), None, Vec::new(), cx)
    }));
    wait_until(cx, "the post lands in the space that was kept", |_| {
        core.runtime()
            .block_on(core.get_space_messages(space_id.clone()))
            .is_ok_and(|m| m.len() == 1)
    });
}

/// **A window that opens while a verdict is outstanding waits for it, and then
/// reports it** — the *after* side of that same segment boundary, where the
/// delete is already travelling and `open` may not proceed blind.
///
/// The store cannot stage this arm: a window opened before the recheck cancels
/// the disposal outright (the test above), so "gated **and** deleted" is
/// unreachable from there — which is itself the point. So the two halves are
/// driven directly: an entity constructed the way `open` constructs one inside
/// the window, and the verdict the disposal's task delivers to it. What must
/// hold is that nothing escapes in either direction — no read answers, no write
/// reaches the database, and when the verdict is *taken* the window says so
/// rather than offering a composer over nothing.
#[gpui::test]
fn a_window_waiting_on_a_verdict_never_writes_and_then_reports_it(cx: &mut TestAppContext) {
    use eidola_gui::space::Space;

    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space_id = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("create space")
        .id;

    // Exactly what `SpacesStore::open` builds when a disposal is in flight.
    let app_core = stores.app_core();
    let space = cx.new(|cx| Space::existing(app_core, space_id.clone(), true, cx));
    assert!(space.read_with(cx, |s, _| s.row_undecided_for_test()));
    assert!(
        !space.read_with(cx, |s, _| s.transcript_visible()),
        "no read has answered, so no composer is minted"
    );

    // A save issued into that window is accepted on the frame and held.
    assert!(space.update(cx, |s, cx| {
        s.submit("The tide is the moon's doing.".into(), None, Vec::new(), cx)
    }));
    settle(cx);
    assert_eq!(
        core.runtime()
            .block_on(core.get_space_messages(space_id.clone()))
            .unwrap()
            .len(),
        0,
        "and nothing reached the database while the verdict was outstanding"
    );

    // The verdict: the space was untouched, and it is gone.
    assert!(
        core.runtime()
            .block_on(core.discard_if_pristine(space_id.clone()))
            .unwrap()
    );
    space.update(cx, |s, cx| {
        s.row_is_gone(
            eidola_app_core::error::AppError::NotConfigured {
                message: "This conversation was empty and untouched, so it was \
                          discarded when its last window closed."
                    .into(),
            },
            cx,
        )
    });
    settle(cx);

    assert!(
        !space.read_with(cx, |s, _| s.row_undecided_for_test()),
        "the verdict settled the gate rather than leaving the window waiting"
    );
    assert!(
        matches!(
            space.read_with(cx, |s, _| s.transcript().clone()),
            eidola_gui::loadable::Loadable::Failed { .. }
        ),
        "the window *reports* the conversation is gone — `Loading` is equally \
         invisible and would be a permanent spinner instead"
    );
    assert!(
        !space.read_with(cx, |s, _| s.transcript_visible()),
        "and it never offers a composer over a space that does not exist"
    );
    assert!(
        !space_row_exists(&core, &space_id),
        "and the held save never resurrected it"
    );
}

/// Give work that is *not* supposed to happen a real chance to happen.
fn settle(cx: &mut TestAppContext) {
    for _ in 0..8 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();
}

/// **A write already on its way to the database outranks the disposal.**
///
/// `bridge` cancels nothing core-side, so a save started a moment before the
/// window closed is still travelling while the disposal reserves the writer. If
/// the disposal gets there first the space is still pristine, so it goes — and
/// the save then lands on nothing, with no window left to say so. That is a
/// reader's words gone, which is the one outcome the whole feature exists to
/// prevent, so the close refuses to dispose over any outstanding write.
///
/// The store's half: it honours the answer it is handed. (The answer itself is
/// taken at the view's release — see `crates/eidola-gui/tests/behavior.rs`,
/// `a_save_still_in_flight_holds_off_the_disposal`.)
#[gpui::test]
fn the_store_never_disposes_over_an_outstanding_write(cx: &mut TestAppContext) {
    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");

    let entity = stores.spaces.update(cx, |s, cx| s.create(cx));
    let space_id = entity.read_with(cx, |s, _| s.id().to_string());
    drop(entity);
    wait_until(cx, "the space's row commits", |_| {
        space_row_exists(&core, &space_id)
    });

    stores
        .spaces
        .update(cx, |s, cx| s.window_closed(space_id.clone(), true, cx));
    settle(cx);

    assert!(
        space_row_exists(&core, &space_id),
        "an untouched space is kept while anything is still writing to it"
    );
    assert!(
        stores
            .spaces
            .read_with(cx, |s, _| s.disposed_for_test().is_empty()),
        "app-core was never asked"
    );
}

/// **A window on a space that turned out not to exist says so — it never
/// spins.**
///
/// The path is ordinary: a Send pressed in the frames after ⌘N waits out the
/// insert, the insert is refused, and the waiter is released with that
/// refusal. Every save-side failure completion then restarts the transcript
/// load — the debt a superseded load owes — and a restart over a settled
/// negative row is the trap: `Failed { prior: None }.to_loading()` is
/// `Loading`, so the reload flips the cell to a spinner, its own gate refuses
/// before anything can resolve it, and the window is left spinning forever
/// with the real error thrown away on the way. So a read against a row known
/// not to exist does not start at all, and the verdict stands.
///
/// The same shape reaches the disposal's verdict, which settles the identical
/// gate; the assertion is on the cell rather than on `transcript_visible`,
/// because `Loading` and `Failed` are both invisible and only one of them is
/// honest.
#[gpui::test]
fn a_refused_creation_after_an_early_send_says_so_instead_of_spinning(cx: &mut TestAppContext) {
    use eidola_gui::loadable::Loadable;

    let (stores, _dir) = backed_stores(cx);

    let space = stores.spaces.update(cx, |s, cx| s.create(cx));
    let space_id = space.read_with(cx, |s, _| s.id().to_string());
    assert!(space.update(cx, |s, cx| {
        s.submit("The tide is the moon's doing.".into(), None, Vec::new(), cx)
    }));

    stores.spaces.update(cx, |s, cx| {
        s.fail_creation_for_test(
            &space_id,
            eidola_app_core::error::AppError::Database {
                message: "disk is full".into(),
            },
            cx,
        )
    });
    settle(cx);

    let cell = space.read_with(cx, |s, _| match s.transcript() {
        Loadable::Failed { error, .. } => format!("failed: {error}"),
        Loadable::Loading => "loading".to_string(),
        Loadable::NotLoaded => "not loaded".to_string(),
        Loadable::Loaded { .. } => "loaded".to_string(),
    });
    assert!(
        cell.starts_with("failed:"),
        "the window holds the refusal, not a spinner — got {cell:?}"
    );
    assert!(
        cell.contains("disk is full"),
        "and it is the creation's own error, not something a reload replaced \
         it with — got {cell:?}"
    );
}

/// **A conversation this window could not read carries its own way back.**
///
/// A failed *initial* transcript load is the one dead end in a space window:
/// there are no posts, so `sync_tail_drafts` mints no composer, and
/// `Space::load_transcript` re-runs only on construction, a bus
/// `Change::Space`, or a mutation's failure exit — none of which the reader can
/// cause with nothing on screen to act on. Recovery meant another writer's
/// invalidation or closing every window on the space.
///
/// Both arms of the Retry are driven against a real core, staged by the id
/// itself: a space that does not exist reads as `NotConfigured`, and creating
/// the row under that same id is what turns the next read into an answer.
///
/// The failing arm is the one with a trap in it — the cell must land back on
/// `Failed`, never on the `Loading` it passes through, or the retry replaces an
/// honest error with a permanent spinner.
#[gpui::test]
fn a_failed_initial_transcript_load_can_be_retried(cx: &mut TestAppContext) {
    use eidola_gui::loadable::Loadable;

    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space_id = eidola_app_core::new_space_id();

    // A window opened on an id naming no row: the initial read refuses.
    let space = stores
        .spaces
        .update(cx, |s, cx| s.open(space_id.clone(), cx));
    wait_until(cx, "the initial transcript read fails", |cx| {
        space.read_with(cx, |s, _| s.transcript_load_failure().is_some())
    });
    assert!(
        !space.read_with(cx, |s, _| s.transcript_visible()),
        "no tree answered, so there is no composer to act from — the dead end"
    );

    // Retry, still with no row behind the id: the error stands, and the cell
    // is settled rather than left spinning.
    space.update(cx, |s, cx| s.retry_transcript_load(cx));
    settle(cx);
    let failure = space.read_with(cx, |s, _| {
        s.transcript_load_failure().map(|e| e.to_string())
    });
    assert!(
        failure
            .as_deref()
            .is_some_and(|e| e.contains("space not found")),
        "a retry that failed keeps the honest error — got {failure:?}"
    );
    assert!(
        !space.read_with(cx, |s, _| matches!(s.transcript(), Loadable::Loading)),
        "and never leaves a spinner behind it"
    );

    // The row appears (another writer, a recovered database) — the same Retry
    // now opens the conversation.
    core.runtime()
        .block_on(core.create_space_with_id(space_id.clone(), None))
        .expect("create the row under the same id");
    space.update(cx, |s, cx| s.retry_transcript_load(cx));
    wait_until(cx, "the retried load answers", |cx| {
        space.read_with(cx, |s, _| s.transcript_visible())
    });
    assert!(
        space.read_with(cx, |s, _| s.transcript_load_failure().is_none()),
        "a load that answered leaves no failure surface standing"
    );
}

/// **A refresh that failed over posts already on screen says so, without taking
/// the posts away.**
///
/// `Failed { prior: Some(..) }` renders as an ordinary conversation — which is
/// the doctrine (a re-fetch must never blank a page) and exactly why silence
/// here is wrong: those posts are as of the last read that succeeded, nothing in
/// this window re-reads on its own, and the composer is not a way to re-run a
/// *read*. So the surface owes the other half, a quiet retry, and this drives
/// the entity behind it.
///
/// Staged on a real core by the id: a space whose row does not exist reads as
/// `NotConfigured`, so the retry can be made to fail over retained posts — the
/// arm with the trap in it, since the cell must come back to `Failed` while
/// still holding them, and must never pass through the valueless `Loading` that
/// would blank the page mid-flight.
#[gpui::test]
fn a_failed_transcript_refresh_keeps_its_posts_and_can_be_retried(cx: &mut TestAppContext) {
    use eidola_gui::loadable::Loadable;

    let (stores, _dir) = backed_stores(cx);
    let core = stores.app_core().expect("backed stores carry a core");
    let space_id = eidola_app_core::new_space_id();

    let space = stores
        .spaces
        .update(cx, |s, cx| s.open(space_id.clone(), cx));
    settle(cx);

    // Posts on screen, and then a refresh over them that failed.
    space.update(cx, |s, cx| {
        s.set_messages_for_test(
            vec![eidola_app_core::SpaceMessage {
                role: "user".into(),
                content: "the question".into(),
            }],
            cx,
        );
        s.fail_transcript_refresh_for_test(cx);
    });
    assert!(
        space.read_with(cx, |s, _| s.transcript_refresh_failure().is_some()),
        "the failed refresh is reported"
    );
    assert!(
        space.read_with(cx, |s, _| s.transcript_load_failure().is_none()),
        "and not as the dead-end surface — there are posts and a composer"
    );
    assert_eq!(
        space.read_with(cx, |s, _| s.messages().len()),
        1,
        "the posts we had stay on screen"
    );

    // Retrying with no row behind the id fails again — and the page must not
    // blank on the way, in flight or at rest.
    space.update(cx, |s, cx| s.retry_transcript_load(cx));
    assert!(
        space.read_with(cx, |s, _| s.transcript_visible()),
        "a retry over retained posts keeps them visible while it runs"
    );
    assert!(
        !space.read_with(cx, |s, _| matches!(s.transcript(), Loadable::Loading)),
        "`Failed {{ prior: Some }}` must go stale, never blank to Loading"
    );
    settle(cx);
    assert!(
        space.read_with(cx, |s, _| s.transcript_refresh_failure().is_some()),
        "a retry that failed again keeps saying so"
    );
    assert_eq!(
        space.read_with(cx, |s, _| s.messages().len()),
        1,
        "still over the posts it had"
    );

    // The row appears; the same retry clears the strip.
    core.runtime()
        .block_on(core.create_space_with_id(space_id.clone(), None))
        .expect("create the row under the same id");
    space.update(cx, |s, cx| s.retry_transcript_load(cx));
    wait_until(cx, "the retried refresh answers", |cx| {
        space.read_with(cx, |s, _| s.transcript_refresh_failure().is_none())
    });
    settle(cx);
    assert!(
        space.read_with(cx, |s, _| s.transcript_refresh_failure().is_none()
            && matches!(s.transcript(), Loadable::Loaded { .. })),
        "a read that answered leaves no failure surface standing"
    );
}
