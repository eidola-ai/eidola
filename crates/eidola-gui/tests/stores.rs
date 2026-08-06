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
