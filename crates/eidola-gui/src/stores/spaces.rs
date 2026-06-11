//! `SpacesStore` — the Library index (the list of non-archived spaces) **and**
//! the per-space entity registry.
//!
//! Per `docs/architecture/state.md` ("Space entities — shared, registried"),
//! this store holds `HashMap<SpaceId, WeakEntity<Space>>`. Opening a space goes
//! through [`SpacesStore::open`], which gets-or-creates: two windows on the
//! same space share **one** `Space` entity (and one transcript load), which is
//! the structural fix for wave-2 bug 4 (a submit/stream in window A appears in
//! window B). [`SpacesStore::blank`] mints an id-less space for ⌘N; the
//! registry adopts it when its first exchange assigns an id (a subscriber on
//! the space's `StreamEnded` event reads its now-present id and keys it).

use std::collections::HashMap;
use std::sync::Arc;

use eidola_app_core::{AppCore, SpaceInfo};
use gpui::{AppContext, Context, Entity, Subscription, Task, WeakEntity};

use crate::bridge::bridge;
use crate::loadable::Loadable;
use crate::space::{Space, SpaceEvent};

pub struct SpacesStore {
    app_core: Option<Arc<AppCore>>,
    /// The Library index (archived excluded), newest activity first.
    index: Loadable<Vec<SpaceInfo>>,
    /// Supersede slot for the index refresh.
    task: Option<Task<()>>,
    /// The per-space entity registry. `WeakEntity` so a dropped window's space
    /// (with no other holder) is collected — `open` prunes dead weaks on miss.
    registry: HashMap<String, WeakEntity<Space>>,
    /// Spaces minted blank (⌘N) that have not yet been adopted into the
    /// registry. Each is held as a weak handle + the subscription that watches
    /// for its first id assignment. On `StreamEnded` the space's now-present
    /// id is read and the entity is moved into `registry`.
    pending_blanks: Vec<(WeakEntity<Space>, Subscription)>,
}

impl SpacesStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            index: Loadable::NotLoaded,
            task: None,
            registry: HashMap::new(),
            pending_blanks: Vec::new(),
        }
    }

    /// A stub store with a fixture listing (tests).
    pub fn stub(spaces: Vec<SpaceInfo>) -> Self {
        Self {
            app_core: None,
            index: if spaces.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(spaces)
            },
            task: None,
            registry: HashMap::new(),
            pending_blanks: Vec::new(),
        }
    }

    /// The current Library index.
    pub fn index(&self) -> &Loadable<Vec<SpaceInfo>> {
        &self.index
    }

    /// The listing as a slice.
    pub fn list(&self) -> &[SpaceInfo] {
        self.index.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    // -- The space-entity registry ----------------------------------------

    /// Open a space by id, getting-or-creating its shared `Space` entity.
    ///
    /// Join-existing: if a live entity is already registered for `id` (another
    /// window opened it), the *same* `Entity<Space>` is returned — both
    /// windows then observe one entity and one transcript load, so a
    /// submit/stream in one appears in the other (wave-2 bug 4). A miss (no
    /// entry, or a collected weak) creates a fresh `Space::existing`, which
    /// kicks off the transcript load once; the second concurrent open joins
    /// that same in-flight load by sharing the entity rather than starting a
    /// duplicate fetch.
    pub fn open(&mut self, id: String, cx: &mut Context<Self>) -> Entity<Space> {
        if let Some(weak) = self.registry.get(&id)
            && let Some(entity) = weak.upgrade()
        {
            return entity;
        }
        let app_core = self.app_core.clone();
        let entity = cx.new(|cx| Space::existing(app_core, id.clone(), cx));
        self.registry.insert(id, entity.downgrade());
        entity
    }

    /// Mint a blank space for ⌘N: id-less, instant, no transcript load. The
    /// registry adopts it once its first exchange persists and assigns an id —
    /// whether that exchange *succeeds* (`StreamEnded`) or *fails after the
    /// space was persisted* (`Failed`, where app-core's `ChatFailed` wrapper let
    /// the space learn its id). Both events populate `Space::id()` before they
    /// fire, so a single `adopt_blank` covers both.
    pub fn blank(&mut self, cx: &mut Context<Self>) -> Entity<Space> {
        let app_core = self.app_core.clone();
        let entity = cx.new(|_| Space::blank(app_core));
        let weak = entity.downgrade();
        let sub = cx.subscribe(&entity, |this, space, event, cx| {
            // `adopt_blank` is a no-op when the id is still `None` (a `Failed`
            // before the space was persisted — e.g. `NoAccount` — never adopts).
            if matches!(event, SpaceEvent::StreamEnded | SpaceEvent::Failed(_)) {
                this.adopt_blank(&space, cx);
            }
        });
        self.pending_blanks.push((weak, sub));
        entity
    }

    /// Adopt a now-id'd blank space into the keyed registry, dropping the
    /// pending-blank bookkeeping for it. Idempotent: re-adoption of an
    /// already-registered id (e.g. multiple `StreamEnded`s) just refreshes the
    /// weak handle.
    fn adopt_blank(&mut self, space: &Entity<Space>, cx: &mut Context<Self>) {
        let Some(id) = space.read(cx).id().map(str::to_string) else {
            return;
        };
        self.registry.insert(id, space.downgrade());
        // Drop the pending-blank entry (and its subscription) for this entity.
        let target = space.downgrade();
        self.pending_blanks
            .retain(|(weak, _)| weak.entity_id() != target.entity_id());
    }

    /// React to a `Change::Space(id)` from the bus by telling the live
    /// registered `Space` (if any) to refresh its transcript. Routed here from
    /// `stores::dispatch_change`.
    pub fn notify_space_changed(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(weak) = self.registry.get(id)
            && let Some(entity) = weak.upgrade()
        {
            entity.update(cx, |space, cx| space.on_space_changed(id, cx));
        }
    }

    // -- The Library index ------------------------------------------------

    /// Refresh the Library index. Fire-and-notify supersede slot.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.index = std::mem::take(&mut self.index).to_loading();
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.list_spaces(false).await }).await;
            let _ = this.update(cx, |this, cx| {
                this.index = std::mem::take(&mut this.index).resolve(result);
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Archive a space: drop the cached row immediately (so the Library
    /// responds without a backend round-trip — and so stub tests exercise the
    /// local path), then archive core-side and let `Change::SpaceIndex` drive
    /// the reconciling refresh. The optimistic removal makes this safe to own
    /// in the store: even if the window closes, the core write completes and
    /// the bus reconciles every other window.
    pub fn archive(&mut self, space_id: String, cx: &mut Context<Self>) {
        // Optimistic local removal — operate on whatever value is present.
        if let Some(list) = self.index.value() {
            let next: Vec<SpaceInfo> = list.iter().filter(|s| s.id != space_id).cloned().collect();
            self.index = Loadable::loaded(next);
        }
        cx.notify();

        let Some(core) = self.app_core.clone() else {
            return;
        };
        // Own the core write in the supersede slot; its completion triggers a
        // reconciling refresh from the bus (`Change::SpaceIndex`). We re-list
        // here too so a stub-less local run reconciles even without the bus.
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, move |c| async move {
                c.archive_space(space_id).await?;
                c.list_spaces(false).await
            })
            .await;
            let _ = this.update(cx, |this, cx| {
                if let Ok(spaces) = result {
                    this.index = Loadable::loaded(spaces);
                }
                this.task = None;
                cx.notify();
            });
        }));
    }
}
