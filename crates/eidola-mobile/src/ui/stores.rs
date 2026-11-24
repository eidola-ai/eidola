//! The domain stores (`crates/eidola-gui/STATE.md`): the Library index and the
//! conversation registry, the account, the wallet and the self-update
//! (`super::account_store`, `super::update_store`), and the bus bridge that
//! keeps them true.
//!
//! Views never call `AppCore`; they call these, and observe them. Every write
//! goes through app-core, which announces each commit on its change bus; the
//! bridge here dispatches each announcement to the store or conversation it
//! concerns, so nothing polls and nothing assumes it is the only writer.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use eidola_app_core::changes::{Change, ChangeEvent};
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, SpaceInfo};
use gpui::{App, AppContext as _, Context, Entity, Subscription, Task, WeakEntity};
use tokio::sync::broadcast::error::RecvError;

use super::account_store::{AccountStore, WalletStore};
use super::conversation::{Conversation, Origin, RowState};
use super::host::Host;
use super::loadable::Loadable;
use super::update_store::UpdateStore;

/// Everything a view is handed.
#[derive(Clone)]
pub struct Stores {
    pub core: Rc<AppCore>,
    pub host: Rc<dyn Host>,
    pub spaces: Entity<SpacesStore>,
    pub account: Entity<AccountStore>,
    pub wallet: Entity<WalletStore>,
    pub updates: Entity<UpdateStore>,
}

impl Stores {
    pub fn new(core: Rc<AppCore>, host: Rc<dyn Host>, cx: &mut App) -> Self {
        let spaces = cx.new(|cx| {
            let mut store = SpacesStore::new(core.clone());
            store.refresh(cx);
            store
        });
        let account = cx.new(|cx| AccountStore::new(core.clone(), cx));
        let wallet = cx.new(|_| WalletStore::new(core.clone()));
        let updates = cx.new(|cx| UpdateStore::new(core.clone(), host.clone(), cx));
        Self {
            core,
            host,
            spaces,
            account,
            wallet,
            updates,
        }
    }
}

/// The Library index and the registry of live conversations.
pub struct SpacesStore {
    core: Rc<AppCore>,
    index: Loadable<Vec<SpaceInfo>>,
    refresh_task: Option<Task<()>>,
    /// Creations in flight, keyed by the id they mint.
    creations: HashMap<String, Task<()>>,
    /// New spaces left before their creation landed: disposed of once it does.
    reap_after_creation: HashSet<String>,
    /// Disposals in flight, keyed by space.
    disposals: HashMap<String, Task<()>>,
    /// One entity per open space, shared by every view of it.
    registry: HashMap<String, WeakEntity<Conversation>>,
    /// Conversations no view shows that still have work in flight, kept alive
    /// until it settles (the observation releases them).
    held: HashMap<String, (Entity<Conversation>, Subscription)>,
}

impl SpacesStore {
    fn new(core: Rc<AppCore>) -> Self {
        Self {
            core,
            index: Loadable::NotLoaded,
            refresh_task: None,
            creations: HashMap::new(),
            reap_after_creation: HashSet::new(),
            disposals: HashMap::new(),
            registry: HashMap::new(),
            held: HashMap::new(),
        }
    }

    pub fn index(&self) -> &Loadable<Vec<SpaceInfo>> {
        &self.index
    }

    /// The listing's row for `id`, when the index has one.
    pub fn info(&self, id: &str) -> Option<&SpaceInfo> {
        self.index.value()?.iter().find(|s| s.id == id)
    }

    /// Re-read the index (supersede: the newest read wins).
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        self.index = std::mem::replace(&mut self.index, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        self.refresh_task = Some(cx.spawn(async move |this, cx| {
            let result = core.list_spaces(false).await;
            this.update(cx, |this, cx| {
                this.refresh_task = None;
                this.index =
                    std::mem::replace(&mut this.index, Loadable::NotLoaded).resolve(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    /// The conversation for `id`: the live entity if one is open, else a new
    /// one. A space whose disposal is being decided opens undecided, and the
    /// disposal settles it.
    pub fn open(&mut self, id: &str, cx: &mut Context<Self>) -> Entity<Conversation> {
        if let Some(live) = self.registry.get(id).and_then(WeakEntity::upgrade) {
            return live;
        }
        let row = if self.disposals.contains_key(id) {
            RowState::Deciding
        } else {
            RowState::Committed
        };
        let core = self.core.clone();
        let id = id.to_string();
        let entity = cx.new(|cx| Conversation::new(core, id.clone(), row, Origin::Listed, cx));
        self.registry.insert(id, entity.downgrade());
        entity
    }

    /// A new conversation: the id is minted here and the entity registered
    /// under it at once (an instant, answered-and-empty page); the row commits
    /// behind it, from the default template. A refused creation leaves no
    /// space, and the entity says so.
    pub fn create(&mut self, cx: &mut Context<Self>) -> Entity<Conversation> {
        let id = eidola_app_core::new_space_id();
        let core = self.core.clone();
        let entity = cx.new(|cx| {
            Conversation::new(
                core.clone(),
                id.clone(),
                RowState::Deciding,
                Origin::Created,
                cx,
            )
        });
        self.registry.insert(id.clone(), entity.downgrade());
        let weak = entity.downgrade();
        let key = id.clone();
        self.creations.insert(
            key.clone(),
            cx.spawn(async move |this, cx| {
                let result = core.create_space_with_id(id.clone(), None).await;
                this.update(cx, |this, cx| {
                    this.creations.remove(&key);
                    match result {
                        Ok(_) => {
                            weak.update(cx, |c, cx| c.row_committed(cx)).ok();
                            if this.reap_after_creation.remove(&key) {
                                this.dispose_if_pristine(key.clone(), cx);
                            }
                        }
                        Err(e) => {
                            this.reap_after_creation.remove(&key);
                            this.registry.remove(&key);
                            weak.update(cx, |c, cx| c.row_is_gone(e, cx)).ok();
                        }
                    }
                    cx.notify();
                })
                .ok();
            }),
        );
        entity
    }

    /// The reader left a conversation's screen.
    ///
    /// **Navigation never cancels domain work**: a conversation still saving,
    /// streaming an answer or planning the replies to one is held here — its
    /// view is gone, and the registry's reference is weak — until that work
    /// settles, so the save's plan is followed, the answer lands with its
    /// recovery state, and a reader who comes back joins the same entity and
    /// its running stream (`open`). Dropping it would cancel exactly the
    /// continuations that do those things.
    ///
    /// A space nobody wrote in goes with its screen — otherwise every "new
    /// conversation" tapped and abandoned stays in the Library — but a write
    /// this app issued outranks the disposal: a busy conversation is never
    /// disposed of, and app-core decides inside the delete whether anything
    /// there is worth keeping.
    pub fn left(&mut self, conversation: &Entity<Conversation>, cx: &mut Context<Self>) {
        let (id, busy, deciding) = {
            let c = conversation.read(cx);
            (c.id().to_string(), c.is_busy(), c.is_deciding())
        };
        if busy {
            let key = id.clone();
            let settled = cx.observe(conversation, move |this, conversation, cx| {
                if !conversation.read(cx).is_busy() {
                    this.held.remove(&key);
                }
            });
            self.held.insert(id, (conversation.clone(), settled));
            return;
        }
        if deciding && self.creations.contains_key(&id) {
            self.reap_after_creation.insert(id);
            return;
        }
        self.dispose_if_pristine(id, cx);
    }

    fn dispose_if_pristine(&mut self, id: String, cx: &mut Context<Self>) {
        if self.disposals.contains_key(&id) {
            return;
        }
        let core = self.core.clone();
        let key = id.clone();
        self.disposals.insert(
            key.clone(),
            cx.spawn(async move |this, cx| {
                let result = core.discard_if_pristine(id).await;
                this.update(cx, |this, cx| {
                    this.disposals.remove(&key);
                    // A view that reopened the space meanwhile is waiting on
                    // this verdict.
                    if let Some(live) = this.registry.get(&key).and_then(WeakEntity::upgrade) {
                        live.update(cx, |c, cx| match result {
                            Ok(true) => c.row_is_gone(
                                AppError::Internal {
                                    message: "the conversation was empty and was removed".into(),
                                },
                                cx,
                            ),
                            _ => c.row_committed(cx),
                        });
                    }
                    if matches!(result, Ok(true)) {
                        this.registry.remove(&key);
                    }
                })
                .ok();
            }),
        );
    }

    /// Whether a conversation no view shows is being kept alive for its work
    /// in flight.
    pub fn is_held(&self, id: &str) -> bool {
        self.held.contains_key(id)
    }

    /// Dispatch one bus announcement.
    fn on_change(&mut self, event: &ChangeEvent, cx: &mut Context<Self>) {
        match &event.change {
            Change::SpaceIndex => self.refresh(cx),
            Change::Space(id) => {
                if let Some(live) = self.registry.get(id).and_then(WeakEntity::upgrade) {
                    live.update(cx, |c, cx| c.on_space_changed(cx));
                }
                // A post moves the listing's order, snippet and title.
                self.refresh(cx);
            }
            // Bylines resolve names at read time.
            Change::Participants => self.reload_live(cx),
            _ => {}
        }
    }

    /// The bus lagged: every read this store or a live conversation holds
    /// may be missing a write.
    fn on_lagged(&mut self, cx: &mut Context<Self>) {
        self.refresh(cx);
        self.reload_live(cx);
    }

    fn reload_live(&mut self, cx: &mut Context<Self>) {
        self.registry.retain(|_, weak| weak.upgrade().is_some());
        let live: Vec<_> = self
            .registry
            .values()
            .filter_map(WeakEntity::upgrade)
            .collect();
        for conversation in live {
            conversation.update(cx, |c, cx| c.on_space_changed(cx));
        }
    }
}

/// Forward the change bus to the stores for the app's lifetime — the one
/// place the core's channels meet gpui, and so its one sanctioned detached
/// task: nothing ever needs to cancel it.
pub fn install_bus_bridge(stores: &Stores, preferred_languages: Vec<String>, cx: &mut App) {
    let mut events = stores.core.subscribe_changes();
    let spaces = stores.spaces.downgrade();
    let account = stores.account.downgrade();
    let wallet = stores.wallet.downgrade();
    let core = stores.core.clone();
    cx.spawn(async move |cx| {
        loop {
            let received = events.recv().await;
            let dispatched = match received {
                // The appearance and language settings live in the config.
                Ok(ChangeEvent {
                    change: Change::Config,
                    ..
                }) => {
                    let config = core.config_state();
                    cx.update(|cx| {
                        super::theme::set_settings(super::theme::settings_of(&config), cx);
                        super::i18n::apply(
                            super::i18n::resolve(config.language.as_deref(), &preferred_languages),
                            cx,
                        );
                    });
                    // The account's credentials live in the config too.
                    account.update(cx, |a, cx| a.on_config_changed(cx))
                }
                Ok(ChangeEvent {
                    change: Change::Backends,
                    ..
                }) => account.update(cx, |a, cx| a.refresh_backends(cx)),
                Ok(ChangeEvent {
                    change: Change::Wallet | Change::Account,
                    ..
                }) => wallet.update(cx, |w, cx| w.on_changed(cx)),
                Ok(event) => spaces.update(cx, |s, cx| s.on_change(&event, cx)),
                Err(RecvError::Lagged(_)) => {
                    account
                        .update(cx, |a, cx| {
                            a.on_config_changed(cx);
                            a.refresh_backends(cx);
                        })
                        .ok();
                    wallet.update(cx, |w, cx| w.on_changed(cx)).ok();
                    spaces.update(cx, |s, cx| s.on_lagged(cx))
                }
                Err(RecvError::Closed) => break,
            };
            if dispatched.is_err() {
                break;
            }
        }
    })
    .detach();
}
