//! Domain stores — the in-memory projection of app-core state, one gpui
//! entity per domain, created at startup and held in `AppGlobal`.
//!
//! This module replaces the old `Core` god-object. See
//! `crates/eidola-gui/STATE.md` ("Domain stores", "Loadable", "Concurrency
//! patterns") for the governing contract. In short:
//!
//! - Each store owns its `Loadable` snapshots, its in-flight `Task` fields
//!   (supersede slots — replacing the field cancels the predecessor), its
//!   subscription to the invalidation bus, and *all* mutations of its domain.
//! - **No shared busy flag.** "In flight" is the presence of a task field /
//!   `Loading` state on the *specific* operation.
//! - **No `.detach()` for domain work.** The one sanctioned exception is the
//!   app-lifetime bus bridge installed at startup (see [`install_bus_bridge`]).
//! - Stores expose `refresh_*` (fire-and-notify; the store owns the slot) and
//!   `request_*` (awaitable; the caller owns the await). Ownership is
//!   cancellation authority, decided per operation.
//!
//! Views never hold an `AppCore` directly: they hold the [`Stores`] bundle (or
//! the individual store entities pulled from it) and call store methods.

pub mod account;
pub mod agents;
pub mod backends;
pub mod config;
pub mod local_models;
pub mod models;
pub mod participants;
pub mod record;
pub mod space_settings;
pub mod spaces;
pub mod templates;
pub mod update;
pub mod wallet;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use eidola_app_core::AppCore;
use eidola_app_core::changes::{Change, ChangeOrigin};
use gpui::{App, AppContext, AsyncApp, Entity};

pub use account::AccountStore;
pub use agents::AgentsStore;
pub use backends::BackendsStore;
pub use config::ConfigStore;
pub use local_models::LocalModelsStore;
pub use models::{BackendCatalog, ModelsStore};
pub use participants::ParticipantsStore;
pub use record::RecordStore;
pub use space_settings::SpaceSettingsStore;
pub use spaces::SpacesStore;
pub use templates::TemplatesStore;
pub use update::UpdateStore;
pub use wallet::WalletStore;

/// The bundle of domain stores, created once at startup and held in
/// `AppGlobal`. Cheaply cloneable (each field is an `Entity` handle / `Arc`),
/// so it is handed to every view's constructor; the view then observes the
/// specific store entities it renders.
///
/// `app_core` is `None` in stub mode (behavior / visual tests), where stores
/// hold fixture `Loadable` values and never drive async work. The `bridge`
/// module's free functions used by chat streaming and the Record reader take
/// this `Arc<AppCore>` directly.
#[derive(Clone)]
pub struct Stores {
    app_core: Option<Arc<AppCore>>,
    pub config: Entity<ConfigStore>,
    pub backends: Entity<BackendsStore>,
    pub models: Entity<ModelsStore>,
    pub local_models: Entity<LocalModelsStore>,
    pub account: Entity<AccountStore>,
    pub wallet: Entity<WalletStore>,
    pub spaces: Entity<SpacesStore>,
    pub update: Entity<UpdateStore>,
    /// Per-space participant membership (the Participants view's data source);
    /// refreshed on `Change::Participants`.
    pub participants: Entity<ParticipantsStore>,
    /// The space-template registry (the Space Templates settings pane);
    /// refreshed on `Change::Templates`.
    pub templates: Entity<TemplatesStore>,
    /// The shared **agent library** (the Agents settings pane, task 36) —
    /// global agents with their notebooks; refreshed on `Change::Participants`,
    /// the same signal `participants` answers with a per-space re-list.
    pub agents: Entity<AgentsStore>,
    /// Per-space settings (cascade limit, router model) — the space
    /// inspector's data source; refreshed on `Change::Space`.
    pub space_settings: Entity<SpaceSettingsStore>,
    /// Bus-relay only — owns no rows. Record listings live in window-scoped
    /// reader entities (`RecordView`), which observe this store to learn
    /// that the local trail grew (see `stores/record.rs`).
    pub record: Entity<RecordStore>,
}

/// Open the real `AppCore` over this machine's config and data directories —
/// the app's one fallible startup step, and deliberately **not** a store
/// concern.
///
/// It is a free function because it is called before there is an `App` to hand
/// stores to: every way this fails is a way the app cannot come up at all
/// (`DatabaseInUse` when another Eidola holds the single-writer database, a
/// directory that will not resolve, a schema version this build refuses), and
/// the caller reports it through `crate::startup` rather than constructing
/// anything. Directory resolution returns the same typed error as everything
/// else here, so one surface renders the whole class.
///
/// **It also starts the sub-space turn driver** — see the call below.
///
/// **It opens the database, not just the core.** `AppCore::new` takes the
/// single-writer lock but fills its database cell lazily, so a schema this build
/// refuses ("delete your dev database") would sail past this gate and surface
/// much later as a pane full of failed refreshes — the honest error, in the one
/// place that cannot act on it. `AppCore::open_database` makes construction and
/// validation one moment; see its docs for why running app-core's startup sweep
/// this early is safe.
pub fn open_app_core() -> Result<Arc<AppCore>, eidola_app_core::error::AppError> {
    let missing = |what: &str| eidola_app_core::error::AppError::Config {
        message: format!("could not determine the Eidola {what} directory for this account"),
    };
    let config_dir = eidola_app_core::config::default_config_path()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .ok_or_else(|| missing("configuration"))?;
    let data_dir = eidola_app_core::config::default_data_dir().ok_or_else(|| missing("data"))?;
    let core = Arc::new(AppCore::new(config_dir, data_dir)?);
    core.runtime().block_on(core.open_database())?;
    // **The app drives delegated conversations; nothing else in it will.** A
    // sub-space has no window, so app-core gives it its turns — but only if
    // somebody starts that, and leaving the start to whoever eventually spawns
    // one would make a room that sits at its brief forever the *default*
    // outcome of forgetting a line. Started here, at the one place a long-lived
    // Eidola builds its core, it cannot be forgotten. It costs an install with
    // no delegated conversations exactly one empty sweep.
    core.start_subspace_driver();
    Ok(core)
}

impl Stores {
    /// Construct the real, backend-backed stores around an already-open
    /// [`AppCore`] (see [`open_app_core`]) and seed the synchronous
    /// `ConfigStore` snapshot; async cells start `NotLoaded` and are filled by
    /// the startup refreshes and the bus.
    pub fn new(app_core: Arc<AppCore>, cx: &mut App) -> Self {
        Self::with_core(Some(app_core), cx)
    }

    /// Stub stores with no backend, for tests. Every store's async methods
    /// become no-ops; tests install fixture state via [`StoresStub`].
    pub fn stub(cx: &mut App) -> Self {
        Self::with_core(None, cx)
    }

    /// Build stub stores from a [`StoresStub`] fixture. The single
    /// replacement for the old `Core::stub()` field-poking: tests describe
    /// the scene declaratively and each store is constructed via its own
    /// stub constructor with no backend.
    pub fn stub_with(fixture: StoresStub, cx: &mut App) -> Self {
        let config = cx.new(|_| {
            let mut store = ConfigStore::stub(fixture.config_state);
            store.set_eidola_trust_for_test(fixture.eidola_trust);
            store
        });
        let backends = cx.new(|_| BackendsStore::stub(fixture.backends));
        // Explicit per-backend catalogs win; otherwise the flat model list
        // becomes the eidola catalog (the pre-backends fixture shape).
        let models = cx.new(|_| match fixture.backend_catalogs {
            Some(catalogs) => ModelsStore::stub_catalogs(catalogs),
            None => ModelsStore::stub(fixture.models),
        });
        let local_models = cx.new(|_| LocalModelsStore::stub(fixture.local_models));
        let account =
            cx.new(|_| AccountStore::stub(fixture.balances, fixture.prices, fixture.subscription));
        let wallet =
            cx.new(|_| WalletStore::stub(fixture.credential_lifecycle, fixture.credentials));
        let spaces = cx.new(|_| SpacesStore::stub(fixture.spaces));
        let update = cx.new(|_| UpdateStore::stub(fixture.update_check, fixture.update_checking));
        let participants = cx.new(|_| ParticipantsStore::stub(fixture.participants));
        let templates = cx.new(|_| TemplatesStore::stub(fixture.templates));
        let agents = cx.new(|_| AgentsStore::stub(fixture.agents));
        let space_settings = cx.new(|_| SpaceSettingsStore::stub(fixture.space_settings));
        let record = cx.new(|_| RecordStore::new());
        Self {
            app_core: None,
            config,
            backends,
            models,
            local_models,
            account,
            wallet,
            spaces,
            update,
            participants,
            templates,
            agents,
            space_settings,
            record,
        }
    }

    /// Build backend-backed stores around an injected `AppCore` — for tests
    /// that need a real (e.g. tempdir + unreachable-URL) backend without
    /// touching the user's real config/data dirs.
    #[doc(hidden)]
    pub fn for_test(app_core: Arc<AppCore>, cx: &mut App) -> Self {
        Self::with_core(Some(app_core), cx)
    }

    fn with_core(app_core: Option<Arc<AppCore>>, cx: &mut App) -> Self {
        let config = cx.new(|_| ConfigStore::new(app_core.clone()));
        let backends = cx.new(|_| BackendsStore::new(app_core.clone()));
        let models = cx.new(|_| ModelsStore::new(app_core.clone()));
        let local_models = cx.new(|_| LocalModelsStore::new(app_core.clone()));
        let account = cx.new(|_| AccountStore::new(app_core.clone()));
        let wallet = cx.new(|_| WalletStore::new(app_core.clone()));
        let spaces = cx.new(|_| SpacesStore::new(app_core.clone()));
        let update = cx.new(|_| UpdateStore::new(app_core.clone()));
        let participants = cx.new(|_| ParticipantsStore::new(app_core.clone()));
        let templates = cx.new(|_| TemplatesStore::new(app_core.clone()));
        let agents = cx.new(|_| AgentsStore::new(app_core.clone()));
        let space_settings = cx.new(|_| SpaceSettingsStore::new(app_core.clone()));
        let record = cx.new(|_| RecordStore::new());
        Self {
            app_core,
            config,
            backends,
            models,
            local_models,
            account,
            wallet,
            spaces,
            update,
            participants,
            templates,
            agents,
            space_settings,
            record,
        }
    }

    /// The underlying `AppCore`, if backed. Used by the `bridge` free
    /// functions (chat streaming, Record reads) that views own directly, and
    /// by views that need to gate "do I have a backend?" — `None` means stub.
    pub fn app_core(&self) -> Option<Arc<AppCore>> {
        self.app_core.clone()
    }
}

/// Declarative fixture for stub stores in tests (`Stores::stub_with`). Each
/// field maps to the corresponding store's stub constructor; default is the
/// empty / not-loaded scene. Replaces the old `Core::stub()` field-poking.
#[derive(Default)]
pub struct StoresStub {
    pub config_state: Option<eidola_app_core::ConfigState>,
    /// The eidola connection + trust bundle (base URL, measurements, hardware
    /// CAs) — read from the `eidola` backend row in production; supplied here
    /// for the General pane's base-URL + trust rows in stub scenes.
    pub eidola_trust: Option<eidola_app_core::EidolaTrust>,
    /// Configured backends. An empty list leaves the store `NotLoaded`,
    /// which reads as "singletons enabled" (the optimistic default).
    pub backends: Vec<eidola_app_core::BackendInfo>,
    /// Flat eidola model list (the pre-backends fixture shape). Ignored
    /// when `backend_catalogs` is set.
    pub models: Vec<eidola_app_core::ModelInfo>,
    /// Explicit per-backend catalogs for multi-backend scenes.
    pub backend_catalogs: Option<Vec<BackendCatalog>>,
    pub local_models: Option<eidola_app_core::LocalModelsState>,
    pub balances: Option<eidola_app_core::BalancesResult>,
    pub prices: Vec<eidola_app_core::PriceInfo>,
    /// The account's subscription standing. `None` leaves the cell
    /// `NotLoaded` — a surface that has not asked yet.
    pub subscription: Option<eidola_app_core::SubscriptionInfo>,
    pub credentials: Vec<eidola_app_core::CredentialInfo>,
    pub credential_lifecycle: Vec<eidola_app_core::CredentialLifecycleInfo>,
    pub spaces: Vec<eidola_app_core::SpaceInfo>,
    pub update_check: Option<eidola_app_core::updates::UpdateCheckSnapshot>,
    pub update_checking: bool,
    /// One space's fixture participant list (the Participants view's scene).
    pub participants: Option<(String, Vec<eidola_app_core::ParticipantInfo>)>,
    /// Fixture space templates (the Space Templates settings pane's scene).
    pub templates: Vec<eidola_app_core::SpaceTemplateInfo>,
    /// Fixture shared agents (the Agents settings pane's scene). `None` leaves
    /// the roster `NotLoaded`; `Some(vec![])` is an empty library that has
    /// answered.
    pub agents: Option<Vec<eidola_app_core::GlobalAgentInfo>>,
    /// One space's fixture settings (the space inspector's scene).
    pub space_settings: Option<(String, eidola_app_core::SpaceSettings)>,
}

/// Install the single app-lifetime bus bridge: a task on `AppCore`'s tokio
/// runtime forwards every [`Change`] through an `mpsc` channel into one gpui
/// main-thread loop, which dispatches to the stores. This is the *only* place
/// tokio receivers touch gpui.
///
/// On `RecvError::Lagged` (a slow consumer fell behind the broadcast capacity)
/// the bridge refreshes *everything* — the doctrine's prescribed response to a
/// dropped change.
///
/// No-op when there is no backend (stub mode).
///
/// Whether **any write this client has issued about `space_id` is still
/// outstanding** — the question the disposal of an untouched space has to ask
/// before it fires (see [`SpacesStore::window_closed`]).
///
/// It lives here rather than on one store because the answer is spread across
/// four owners and no one of them can see the others: the conversation's own
/// saves and turns are the `Space` entity's (`post_runner`, `streams`,
/// `turn_runners`), a rename or an archive is `SpacesStore`'s, the cascade
/// limit and the router are `SpaceSettingsStore`'s, and the roster — adds,
/// invites, overrides, removals, a share — is `ParticipantsStore`'s. A `bridge`
/// call outlives the gpui task that issued it, so every one of these can be
/// travelling to the database at the moment a window closes.
///
/// **Asked at the close, while everything is still alive to answer.** The
/// entity dies with its last window, so afterwards there is nothing left to
/// ask — which is exactly why the disposal cannot ask for itself. A stale
/// `true` only keeps a space; a stale `false` cannot happen, because nothing
/// can begin writing to a space whose last window has gone.
pub fn space_writes_in_flight(stores: &Stores, space_id: &str, cx: &App) -> bool {
    stores.spaces.read(cx).writes_in_flight(space_id)
        || stores.space_settings.read(cx).writes_in_flight(space_id)
        || stores.participants.read(cx).writes_in_flight(space_id)
}

/// Returns a [`BusBridge`] handle so the quit path can **stop the dispatch
/// loop** before anything else — see [`BusBridge::quiesce`].
pub fn install_bus_bridge(stores: &Stores, cx: &mut App) -> BusBridge {
    let bridge = BusBridge {
        quiesced: Arc::new(AtomicBool::new(false)),
    };
    let Some(app_core) = stores.app_core.clone() else {
        return bridge;
    };

    // The tokio side: a broadcast receiver feeding a gpui-side mpsc. The
    // forwarding task lives on the core runtime; it ends when the unbounded
    // sender is dropped, which happens when the gpui loop below ends.
    let mut bus = app_core.subscribe_changes();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<BridgeEvent>();
    app_core.runtime().handle().clone().spawn(async move {
        loop {
            match bus.recv().await {
                Ok(event) => {
                    if tx
                        .send(BridgeEvent::Change(event.change, event.origin))
                        .is_err()
                    {
                        break; // gpui loop gone — app shutting down
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    if tx.send(BridgeEvent::Lagged).is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    let stores = stores.clone();
    // App-lifetime task: it lives for the whole process and there is nothing
    // to cancel it against, so `.detach()` is the *sanctioned* exception to
    // the no-detach rule (see `crates/eidola-gui/STATE.md` principle 3 and the
    // stores module docs). Every other task in the app is owned by an entity
    // field with replace-cancels semantics.
    let quiesced = bridge.quiesced.clone();
    let task: gpui::Task<()> = cx.spawn(async move |cx: &mut AsyncApp| {
        while let Some(event) = rx.recv().await {
            // The quit gate. Checked on this side — the *receiver* — because
            // this is the single place every `Change` in the process enters
            // gpui, and therefore the only place where "no dispatch after
            // quit begins" can be stated once instead of re-proved for each
            // emitter (see [`BusBridge::quiesce`]).
            if quiesced.load(Ordering::SeqCst) {
                break;
            }
            // `AsyncApp::update` here yields `()` (the dispatch returns unit);
            // ignore via a statement-position call so the loop keeps draining.
            cx.update(|cx| match event {
                BridgeEvent::Change(change, origin) => dispatch_change(&stores, change, origin, cx),
                BridgeEvent::Lagged => refresh_everything(&stores, cx),
            });
        }
    });
    task.detach();
    bridge
}

/// Handle to the app-lifetime bus bridge, held so the quit path can stop it.
///
/// **Why the seam is here and not on each emitter.** A `Change` dispatched
/// after `App::shutdown` has set `quitting` reaches a store's `refresh` →
/// `cx.spawn` → gpui's "Can't spawn on main thread after on_app_quit" panic.
/// Silencing the engine drain fixed one emitter; silencing the engine
/// supervisors fixed a second; an in-flight model *download* reporting
/// progress during the quit's grace window would have been a third, and every
/// future emitter a fresh one. The bridge is the one door all of them come
/// through, so closing it is the cure for the class rather than for its
/// current members.
///
/// **Layering.** This handle is the GUI's protection and covers *every*
/// emitter, present and future. app-core keeps its own latch-gating of the
/// quit-time drain and supervisor arms for a different reason and a different
/// consumer: the CLI embeds `AppCore` with no gpui at all, and the bus is a
/// documented contract there (`tests/bus.rs` enumerates every exit point's
/// emissions) — a teardown nobody is expected to observe should not announce
/// itself on it. Neither layer relies on the other.
#[derive(Clone)]
pub struct BusBridge {
    quiesced: Arc<AtomicBool>,
}

impl BusBridge {
    /// Stop dispatching invalidations into gpui. One-way, and safe to call
    /// when no bridge was installed (stub mode).
    pub fn quiesce(&self) {
        self.quiesced.store(true, Ordering::SeqCst);
    }

    #[doc(hidden)]
    pub fn is_quiesced(&self) -> bool {
        self.quiesced.load(Ordering::SeqCst)
    }
}

enum BridgeEvent {
    Change(Change, ChangeOrigin),
    Lagged,
}

/// Test seam: drive the bridge's dispatch logic directly, without the
/// tokio→gpui plumbing. Lets tests assert "a `Change::X` refreshes store X"
/// deterministically (the live plumbing's timing is exercised by the app at
/// runtime, not by a test). `Lagged` is modelled by passing `None`.
///
/// The origin defaults to [`ChangeOrigin::Caller`] — what every write made on
/// a consumer's own call path carries; [`dispatch_change_event_for_test`] is
/// the seam for the unattended half.
#[doc(hidden)]
pub fn dispatch_change_for_test(stores: &Stores, change: Option<Change>, cx: &mut App) {
    dispatch_change_event_for_test(stores, change, ChangeOrigin::Caller, cx)
}

/// [`dispatch_change_for_test`] naming the origin the change was emitted
/// under — the seam for asserting what a busy surface does with a write
/// nobody is waiting on.
#[doc(hidden)]
pub fn dispatch_change_event_for_test(
    stores: &Stores,
    change: Option<Change>,
    origin: ChangeOrigin,
    cx: &mut App,
) {
    match change {
        Some(change) => dispatch_change(stores, change, origin, cx),
        None => refresh_everything(stores, cx),
    }
}

/// Route one [`Change`] to the store(s) that own the affected domain.
fn dispatch_change(stores: &Stores, change: Change, origin: ChangeOrigin, cx: &mut App) {
    match change {
        Change::Config => {
            stores.config.update(cx, |s, cx| s.refresh(cx));
            // A base-URL flip invalidates the model list (different upstream).
            stores.models.update(cx, |s, cx| s.refresh(cx));
        }
        Change::Account => {
            stores.account.update(cx, |s, cx| s.refresh_balances(cx));
        }
        Change::Wallet => {
            stores.wallet.update(cx, |s, cx| s.refresh(cx));
        }
        Change::SpaceIndex => {
            stores.spaces.update(cx, |s, cx| s.refresh(cx));
        }
        // A per-space message change (e.g. a CLI write to the same space, in
        // process) is routed to the live registered `Space` entity, which
        // refreshes its own transcript. The listing-level signal is
        // `SpaceIndex` (above).
        // A per-space message change (e.g. a CLI write to the same space, in
        // process) is routed to the live registered `Space` entity, which
        // refreshes its own transcript. The same signal carries a space's own
        // settings (cascade limit / router model), so an open inspector — and
        // only an open one, since the store re-reads just what it has cached —
        // re-reads them too.
        Change::Space(id) => {
            stores
                .spaces
                .update(cx, |s, cx| s.notify_space_changed(&id, origin, cx));
            stores
                .space_settings
                .update(cx, |s, cx| s.refresh_if_cached(&id, cx));
        }
        // Record listings are window-scoped reader entities; no global store
        // owns their rows. The RecordStore is the bus seam those readers
        // observe: bumping its epoch lets every open Record window mark
        // itself stale and surface the "new entries — refresh" affordance.
        Change::Record => {
            stores.record.update(cx, |s, cx| s.notify_changed(cx));
        }
        Change::UpdateState => {
            stores.update.update(cx, |s, cx| s.refresh(cx));
        }
        // Downloads progressing, engines loading/unloading, deletions — the
        // whole local-inference domain re-snapshots from one signal.
        Change::LocalModels => {
            stores.local_models.update(cx, |s, cx| s.refresh(cx));
        }
        // The set of configured destinations changed: re-snapshot the
        // registry, the per-backend model catalogs, and the engine domain
        // (a llamacpp backend may have appeared/vanished). The eidola row also
        // carries the connection + trust bundle (base URL / measurements /
        // hardware CAs), so refresh the config store's `EidolaTrust` snapshot
        // too — the base-URL editor and trust rows live in Settings → Backends
        // → Eidola and must reflect another window's write.
        Change::Backends => {
            stores.config.update(cx, |s, cx| s.refresh(cx));
            stores.backends.update(cx, |s, cx| s.refresh(cx));
            stores.models.update(cx, |s, cx| s.refresh(cx));
            stores.local_models.update(cx, |s, cx| s.refresh(cx));
        }
        // Participants v1 domains. A template change can also move the resolved
        // default model (the default template's agent), so refresh the config
        // store's cached `default_model` alongside the templates registry.
        Change::Templates => {
            stores.config.update(cx, |s, cx| s.refresh(cx));
            stores.templates.update(cx, |s, cx| s.refresh(cx));
        }
        // A per-space participant change carries no id, so re-read every cached
        // space's membership.
        //
        // It also reaches the **templates** registry, which is an amendment to
        // the 1:1 variant↔store rule's letter in the service of its spirit: a
        // `SpaceTemplateInfo` embeds its referenced globals' *effective* config
        // (`SpaceTemplateInfo::referenced`), so an "edit everywhere" of a shared
        // participant — which emits only `Change::Participants` — moves what a
        // cached template snapshot says. The store reads participant config now,
        // so it must hear about participant changes. (`Change::Config` already
        // fans out to two stores for the same kind of reason.)
        // The **agent library** answers the same signal with its own listing:
        // a promotion adds a row, a retirement removes one, and an "edit
        // everywhere" moves what a row says. Two domains, one invalidation —
        // the dispatcher fans it out rather than either store polling.
        // The **live transcripts** answer it too, and for the same reason one
        // domain over: a post's byline and a reference edge's carried author
        // identity are resolved by `get_space_tree`'s joins at read time and
        // never re-derived, so a rename left every open window naming the
        // author as they were when it loaded. See
        // `SpacesStore::notify_participants_changed`.
        Change::Participants => {
            stores.participants.update(cx, |s, cx| s.refresh_all(cx));
            stores.templates.update(cx, |s, cx| s.refresh(cx));
            stores.agents.update(cx, |s, cx| s.refresh(cx));
            stores
                .spaces
                .update(cx, |s, cx| s.notify_participants_changed(cx));
        }
    }
}

/// Refresh every store. The `Lagged` response — we missed at least one change,
/// so re-read everything we care about.
fn refresh_everything(stores: &Stores, cx: &mut App) {
    stores.config.update(cx, |s, cx| s.refresh(cx));
    stores.backends.update(cx, |s, cx| s.refresh(cx));
    stores.models.update(cx, |s, cx| s.refresh(cx));
    stores.local_models.update(cx, |s, cx| s.refresh(cx));
    stores.account.update(cx, |s, cx| {
        s.refresh_balances(cx);
        s.refresh_prices(cx);
    });
    stores.wallet.update(cx, |s, cx| s.refresh(cx));
    // The index *and* the live conversations. A store refresh re-reads only
    // the Library listing, so before this a lag that swallowed a rename left
    // every already-open transcript naming its authors as it did when it
    // loaded — with the incoming-reference and trace caches stale beside it —
    // while this function claimed to refresh all state (Codex review, PR #292).
    stores.spaces.update(cx, |s, cx| {
        s.refresh(cx);
        s.notify_lagged(cx);
    });
    stores.update.update(cx, |s, cx| s.refresh(cx));
    stores.templates.update(cx, |s, cx| s.refresh(cx));
    stores.participants.update(cx, |s, cx| s.refresh_all(cx));
    stores.agents.update(cx, |s, cx| s.refresh(cx));
    stores.space_settings.update(cx, |s, cx| s.refresh_all(cx));
    // A dropped change may have been a Record write — let open Record
    // windows mark themselves stale.
    stores.record.update(cx, |s, cx| s.notify_changed(cx));
}

/// Settle a write-through mutation's two outcomes — the write's, and the
/// re-list's — into the domain cell and the op-error slot. Returns the error to
/// record (`None` = the write succeeded); the caller owns where that lives
/// (one slot in `TemplatesStore`, keyed per space in `ParticipantsStore`).
///
/// **The re-list always resolves the cell**, success or failure. That is the
/// whole rule, and it is not tidiness: a mutation *cancels* the in-flight
/// refresh when it takes over the read (see either store's
/// `write_then_relist`), and that refresh may already have moved the cell to
/// `Loading` — or to `Loaded { stale }` over prior data. Its own re-list is
/// then the only thing left that can resolve it, so a re-list error that was
/// merely dropped stranded the cell: a spinner with **no live task behind it**
/// (the `Loadable` invariant inverted), rendering as a plausible-empty registry
/// that nothing would ever correct until an unrelated `Change` happened along.
/// `Loadable::resolve` lands it in `Failed { error, prior }` instead — which
/// keeps prior data visible (STATE.md's "Failed is not empty") and gives the
/// view its honest "couldn't refresh — retry" door.
///
/// **The write's error still wins the op-error slot.** The two errors answer
/// different questions ("your edit was refused" vs "we couldn't re-read the
/// list"), so they occupy different places rather than overwriting each other.
pub(crate) fn settle_mutation<T>(
    cell: &mut crate::loadable::Loadable<T>,
    list: Result<T, eidola_app_core::error::AppError>,
    op: Result<(), String>,
) -> Option<String>
where
    T: Default,
{
    *cell = std::mem::take(cell).resolve(list);
    op.err()
}

#[cfg(test)]
mod tests {
    use super::settle_mutation;
    use crate::loadable::Loadable;
    use eidola_app_core::error::AppError;

    fn err(message: &str) -> AppError {
        AppError::Internal {
            message: message.to_string(),
        }
    }

    #[test]
    fn a_successful_relist_loads_the_cell_and_records_no_error() {
        let mut cell: Loadable<Vec<u8>> = Loadable::NotLoaded.to_loading();
        let recorded = settle_mutation(&mut cell, Ok(vec![1, 2]), Ok(()));
        assert_eq!(recorded, None);
        assert_eq!(cell.value(), Some(&vec![1, 2]));
        assert!(!cell.is_stale(), "a settled re-list is fresh, not stale");
    }

    #[test]
    fn a_failed_write_still_applies_its_relist() {
        // The partial-failure case the unconditional re-list exists for: the
        // listing must show what was in fact created, beside the error.
        let mut cell: Loadable<Vec<u8>> = Loadable::loaded(vec![1]);
        let recorded = settle_mutation(&mut cell, Ok(vec![1, 2]), Err("refused".into()));
        assert_eq!(recorded.as_deref(), Some("refused"));
        assert_eq!(cell.value(), Some(&vec![1, 2]));
    }

    #[test]
    fn a_failed_relist_resolves_a_loading_cell_instead_of_stranding_it() {
        // The strand: a cancelled refresh left the cell `Loading` with no task
        // behind it, and this re-list is the only thing that can still resolve
        // it. Dropping its error left a permanent spinner-shaped empty page.
        let mut cell: Loadable<Vec<u8>> = Loadable::NotLoaded.to_loading();
        assert!(
            cell.is_loading(),
            "precondition: the cancelled refresh's cell"
        );
        let recorded =
            settle_mutation(&mut cell, Err(err("db read failed")), Err("refused".into()));
        assert!(!cell.is_loading(), "the cell must never be left Loading");
        assert!(cell.error().is_some(), "the read failure is visible");
        assert_eq!(
            recorded.as_deref(),
            Some("refused"),
            "the write's error still wins the op-error slot"
        );
    }

    #[test]
    fn a_failed_relist_keeps_prior_data_visible() {
        let mut cell: Loadable<Vec<u8>> = Loadable::loaded(vec![7]).to_loading();
        let recorded = settle_mutation(&mut cell, Err(err("db read failed")), Ok(()));
        assert_eq!(recorded, None);
        assert_eq!(
            cell.value(),
            Some(&vec![7]),
            "Failed is not empty — the prior snapshot stays on screen"
        );
        assert!(cell.error().is_some());
    }
}
