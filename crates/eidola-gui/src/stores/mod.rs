//! Domain stores — the in-memory projection of app-core state, one gpui
//! entity per domain, created at startup and held in `AppGlobal`.
//!
//! This module replaces the old `Core` god-object. See
//! `docs/architecture/state.md` ("Domain stores", "Loadable", "Concurrency
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
pub mod config;
pub mod models;
pub mod record;
pub mod spaces;
pub mod update;
pub mod wallet;

use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::changes::Change;
use gpui::{App, AppContext, AsyncApp, Entity};

pub use account::AccountStore;
pub use config::ConfigStore;
pub use models::ModelsStore;
pub use record::RecordStore;
pub use spaces::SpacesStore;
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
    pub models: Entity<ModelsStore>,
    pub account: Entity<AccountStore>,
    pub wallet: Entity<WalletStore>,
    pub spaces: Entity<SpacesStore>,
    pub update: Entity<UpdateStore>,
    /// Bus-relay only — owns no rows. Record listings live in window-scoped
    /// reader entities (`RecordView`), which observe this store to learn
    /// that the local trail grew (see `stores/record.rs`).
    pub record: Entity<RecordStore>,
}

impl Stores {
    /// Construct the real, backend-backed stores. Creates one `AppCore`
    /// (its own tokio runtime) and seeds the synchronous `ConfigStore`
    /// snapshot; async cells start `NotLoaded` and are filled by the
    /// startup refreshes and the bus.
    pub fn new(cx: &mut App) -> Self {
        let config_dir = eidola_app_core::config::default_config_path()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .expect("could not determine eidola config directory");
        let data_dir = eidola_app_core::config::default_data_dir()
            .expect("could not determine eidola data directory");

        let app_core = Arc::new(AppCore::new(config_dir, data_dir));
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
        let config = cx.new(|_| ConfigStore::stub(fixture.config_state));
        let models = cx.new(|_| ModelsStore::stub(fixture.models));
        let account = cx.new(|_| AccountStore::stub(fixture.balances, fixture.prices));
        let wallet =
            cx.new(|_| WalletStore::stub(fixture.credential_lifecycle, fixture.credentials));
        let spaces = cx.new(|_| SpacesStore::stub(fixture.spaces));
        let update = cx.new(|_| UpdateStore::stub(fixture.update_check, fixture.update_checking));
        let record = cx.new(|_| RecordStore::new());
        Self {
            app_core: None,
            config,
            models,
            account,
            wallet,
            spaces,
            update,
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
        let models = cx.new(|_| ModelsStore::new(app_core.clone()));
        let account = cx.new(|_| AccountStore::new(app_core.clone()));
        let wallet = cx.new(|_| WalletStore::new(app_core.clone()));
        let spaces = cx.new(|_| SpacesStore::new(app_core.clone()));
        let update = cx.new(|_| UpdateStore::new(app_core.clone()));
        let record = cx.new(|_| RecordStore::new());
        Self {
            app_core,
            config,
            models,
            account,
            wallet,
            spaces,
            update,
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
    pub models: Vec<eidola_app_core::ModelInfo>,
    pub balances: Option<eidola_app_core::BalancesResult>,
    pub prices: Vec<eidola_app_core::PriceInfo>,
    pub credentials: Vec<eidola_app_core::CredentialInfo>,
    pub credential_lifecycle: Vec<eidola_app_core::CredentialLifecycleInfo>,
    pub spaces: Vec<eidola_app_core::SpaceInfo>,
    pub update_check: Option<eidola_app_core::updates::UpdateCheckSnapshot>,
    pub update_checking: bool,
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
pub fn install_bus_bridge(stores: &Stores, cx: &mut App) {
    let Some(app_core) = stores.app_core.clone() else {
        return;
    };

    // The tokio side: a broadcast receiver feeding a gpui-side mpsc. The
    // forwarding task lives on the core runtime; it ends when the unbounded
    // sender is dropped, which happens when the gpui loop below ends.
    let mut bus = app_core.subscribe_changes();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<BridgeEvent>();
    app_core.runtime().handle().clone().spawn(async move {
        loop {
            match bus.recv().await {
                Ok(change) => {
                    if tx.send(BridgeEvent::Change(change)).is_err() {
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
    // the no-detach rule (see `docs/architecture/state.md` principle 3 and the
    // stores module docs). Every other task in the app is owned by an entity
    // field with replace-cancels semantics.
    let task: gpui::Task<()> = cx.spawn(async move |cx: &mut AsyncApp| {
        while let Some(event) = rx.recv().await {
            // `AsyncApp::update` here yields `()` (the dispatch returns unit);
            // ignore via a statement-position call so the loop keeps draining.
            cx.update(|cx| match event {
                BridgeEvent::Change(change) => dispatch_change(&stores, change, cx),
                BridgeEvent::Lagged => refresh_everything(&stores, cx),
            });
        }
    });
    task.detach();
}

enum BridgeEvent {
    Change(Change),
    Lagged,
}

/// Test seam: drive the bridge's dispatch logic directly, without the
/// tokio→gpui plumbing. Lets tests assert "a `Change::X` refreshes store X"
/// deterministically (the live plumbing's timing is exercised by the app at
/// runtime, not by a test). `Lagged` is modelled by passing `None`.
#[doc(hidden)]
pub fn dispatch_change_for_test(stores: &Stores, change: Option<Change>, cx: &mut App) {
    match change {
        Some(change) => dispatch_change(stores, change, cx),
        None => refresh_everything(stores, cx),
    }
}

/// Route one [`Change`] to the store(s) that own the affected domain.
fn dispatch_change(stores: &Stores, change: Change, cx: &mut App) {
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
        Change::Space(id) => {
            stores
                .spaces
                .update(cx, |s, cx| s.notify_space_changed(&id, cx));
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
    }
}

/// Refresh every store. The `Lagged` response — we missed at least one change,
/// so re-read everything we care about.
fn refresh_everything(stores: &Stores, cx: &mut App) {
    stores.config.update(cx, |s, cx| s.refresh(cx));
    stores.models.update(cx, |s, cx| s.refresh(cx));
    stores.account.update(cx, |s, cx| {
        s.refresh_balances(cx);
        s.refresh_prices(cx);
    });
    stores.wallet.update(cx, |s, cx| s.refresh(cx));
    stores.spaces.update(cx, |s, cx| s.refresh(cx));
    stores.update.update(cx, |s, cx| s.refresh(cx));
    // A dropped change may have been a Record write — let open Record
    // windows mark themselves stale.
    stores.record.update(cx, |s, cx| s.notify_changed(cx));
}
