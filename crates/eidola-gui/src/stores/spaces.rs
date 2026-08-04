//! `SpacesStore` — the Library index (the list of non-archived spaces) **and**
//! the per-space entity registry.
//!
//! Per `crates/eidola-gui/STATE.md` ("Space entities — shared, registried"),
//! this store holds `HashMap<SpaceId, WeakEntity<Space>>`. Opening a space goes
//! through [`SpacesStore::open`], which gets-or-creates: two windows on the
//! same space share **one** `Space` entity (and one transcript load), which is
//! the structural fix for wave-2 bug 4 (a submit/stream in window A appears in
//! window B). [`SpacesStore::blank`] mints an id-less space for ⌘N; the
//! registry adopts it when its first exchange assigns an id (a subscriber on
//! the space's `StreamEnded` event reads its now-present id and keys it).
//!
//! **The index's mutations follow the write-through rules** the other stores
//! do (`crates/eidola-gui/STATE.md` → "Concurrency patterns"): rename and
//! archive edit the cached row optimistically so the Library answers without a
//! round trip, then compose `write; re-list` in a slot of their **own** — never
//! the refresh's, because a rename emits the very `Change::SpaceIndex` that
//! drives the refresh, and a shared slot would let it cancel the write's own
//! completion (the core write still runs; the continuation carrying its refusal
//! and its re-list never does — a refused write indistinguishable from a
//! successful one). The re-list runs on **every** exit, so the optimism never
//! outlives the round trip: a refused write is reconciled back to what the
//! database holds, with its refusal in [`SpacesStore::op_error`].

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use eidola_app_core::{AppCore, SpaceInfo};
use gpui::{AppContext, Context, Entity, Subscription, Task, WeakEntity};

use crate::bridge::bridge;
use crate::loadable::Loadable;
use crate::space::{Space, SpaceEvent};
use crate::stores::Stores;

/// A refused space operation: the sentence to show, plus the space it was about
/// (`None` for a create — there is no space yet).
///
/// The tag is what keeps one honest banner per window from lying: the Library
/// is the window over the whole index and shows any refusal, while a per-space
/// surface (the space inspector) shows only its own — the index is a store-wide
/// snapshot, so an untagged slot would put one space's refusal under another
/// space's title field.
struct SpaceOpError {
    space_id: Option<String>,
    message: String,
}

pub struct SpacesStore {
    app_core: Option<Arc<AppCore>>,
    /// The Library index (archived excluded), newest activity first.
    index: Loadable<Vec<SpaceInfo>>,
    /// Supersede slot for the index **refresh** — deliberately not the
    /// mutation's (see the module docs).
    task: Option<Task<()>>,
    /// Supersede slot for an index **mutation** (rename / archive), each
    /// composing `write; re-list`.
    op_task: Option<Task<()>>,
    /// A refresh signalled while a mutation held the read: deferred to that
    /// mutation's completion, where it can only be fresher.
    refresh_pending: bool,
    /// The per-space entity registry. `WeakEntity` so a dropped window's space
    /// (with no other holder) is collected — `open` prunes dead weaks on miss.
    registry: HashMap<String, WeakEntity<Space>>,
    /// Spaces minted blank (⌘N) that have not yet been adopted into the
    /// registry. Each is held as a weak handle + the subscription that watches
    /// for its first id assignment. On `StreamEnded` the space's now-present
    /// id is read and the entity is moved into `registry`.
    pending_blanks: Vec<(WeakEntity<Space>, Subscription)>,
    /// In-flight "New Space from Template" ops, keyed by a monotonic id so each
    /// activation is **independent** and runs to completion (STATE.md "keyed
    /// per-key work") — a supersede slot would let a superseded result, whose
    /// core-side space is already committed, strand an empty space with no
    /// window. Each task removes its own key when it finishes.
    create_ops: HashMap<u64, Task<()>>,
    next_create_op: u64,
    /// The last refused space operation — a create, a rename, or an archive.
    /// One slot, because the index it is about is one store-wide snapshot: the
    /// Library shows it as its single banner and the inspector shows it only
    /// when it is tagged with the space that inspector is looking at.
    op_error: Option<SpaceOpError>,
}

impl SpacesStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            index: Loadable::NotLoaded,
            task: None,
            op_task: None,
            refresh_pending: false,
            registry: HashMap::new(),
            pending_blanks: Vec::new(),
            create_ops: HashMap::new(),
            next_create_op: 0,
            op_error: None,
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
            op_task: None,
            refresh_pending: false,
            registry: HashMap::new(),
            pending_blanks: Vec::new(),
            create_ops: HashMap::new(),
            next_create_op: 0,
            op_error: None,
        }
    }

    /// Test seam: apply the settle a mutation's completion applies — the
    /// re-list's listing, plus the write's refusal when it was refused —
    /// without a backend to fail against. It runs the *production*
    /// [`crate::stores::settle_mutation`], so a view test driven through it is
    /// exercising the reconciliation shape the real path takes.
    #[doc(hidden)]
    pub fn settle_for_test(
        &mut self,
        space_id: Option<String>,
        listing: Vec<SpaceInfo>,
        refusal: Option<&str>,
        cx: &mut Context<Self>,
    ) {
        let op = refusal.map_or(Ok(()), |r| Err(r.to_string()));
        if let Some(message) = crate::stores::settle_mutation(&mut self.index, Ok(listing), op) {
            self.op_error = Some(SpaceOpError { space_id, message });
        }
        cx.notify();
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
    ///
    /// **Every** live space also drops its cached incoming-reference indexes,
    /// not just the changed one: a quote written in space B changes what space
    /// A's posts should highlight, and the bus event carries only the written
    /// space's id. The indexes are re-fetched lazily per rendered post, so
    /// this costs a query per *visible* post at most.
    pub fn notify_space_changed(&mut self, id: &str, cx: &mut Context<Self>) {
        for weak in self.registry.values() {
            if let Some(entity) = weak.upgrade() {
                entity.update(cx, |space, cx| space.invalidate_incoming_references(cx));
            }
        }
        if let Some(weak) = self.registry.get(id)
            && let Some(entity) = weak.upgrade()
        {
            entity.update(cx, |space, cx| {
                // Trace disclosures are space-local (a turn's rounds are
                // written in the space that ran them), so only the changed
                // space drops its index.
                space.invalidate_traces(cx);
                space.on_space_changed(id, cx);
            });
        }
    }

    // -- Refused operations -------------------------------------------------

    /// The last refused space operation, whatever it was about — the Library's
    /// banner, which is the window over the whole index.
    pub fn op_error(&self) -> Option<&str> {
        self.op_error.as_ref().map(|e| e.message.as_str())
    }

    /// The last refusal **about `space_id`** — what a per-space surface (the
    /// space inspector) shows, so a rename refused in one space never surfaces
    /// under another space's title field.
    pub fn op_error_for(&self, space_id: &str) -> Option<&str> {
        self.op_error
            .as_ref()
            .filter(|e| e.space_id.as_deref() == Some(space_id))
            .map(|e| e.message.as_str())
    }

    pub fn clear_op_error(&mut self, cx: &mut Context<Self>) {
        if self.op_error.take().is_some() {
            cx.notify();
        }
    }

    // -- New Space from Template ------------------------------------------

    /// Instantiate a **specific** template into a new space and open its window.
    /// The op is **owned** here in a keyed task slot (never `.detach()`ed) so it
    /// runs to completion regardless of any window, and each activation is
    /// independent (no supersede) so a committed space always gets its window —
    /// no stranding. A failure is surfaced in [`SpacesStore::op_error`] (the
    /// Library banner), not silently discarded. `stores` is passed through only to open the
    /// resulting window (`open_space_window` needs the bundle); it is not
    /// retained.
    pub fn create_from_template(
        &mut self,
        template_id: String,
        stores: Stores,
        cx: &mut Context<Self>,
    ) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.op_error = None;
        let key = self.next_create_op;
        self.next_create_op += 1;
        self.create_ops.insert(
            key,
            cx.spawn(async move |this, cx| {
                let result = bridge(core, move |c| async move {
                    c.create_space_from_template(template_id, None).await
                })
                .await;
                match result {
                    Ok(space) => {
                        // Open the window in the App context; the durable space
                        // already committed (and emitted `Change::SpaceIndex`),
                        // so even if this update is missed the Library reflects it.
                        cx.update(|cx| {
                            crate::open_space_window(cx, stores.clone(), space.id);
                        });
                    }
                    Err(e) => {
                        let _ = this.update(cx, |this, cx| {
                            this.op_error = Some(SpaceOpError {
                                space_id: None,
                                message: format!("Couldn't create the space: {e}"),
                            });
                            cx.notify();
                        });
                    }
                }
                let _ = this.update(cx, |this, _| {
                    this.create_ops.remove(&key);
                });
            }),
        );
        cx.notify();
    }

    // -- The Library index ------------------------------------------------

    /// Refresh the Library index. Fire-and-notify supersede slot.
    ///
    /// A refresh signalled while a mutation is in flight is **deferred** to its
    /// completion rather than started beside it: the mutation has taken over
    /// the read for its duration (see [`Self::write_then_relist`]), and a read
    /// that resolves after the mutation's own re-list can only be fresher than
    /// one racing it.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        if self.op_task.is_some() {
            self.refresh_pending = true;
            return;
        }
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

    /// The write-through shape for an index mutation: run `op`, then re-list —
    /// on **every** exit, failure included.
    ///
    /// **The mutation takes over the read.** It drops any in-flight refresh,
    /// which may have been issued *before* this write and would re-stale the
    /// snapshot by resolving after it; cancelling another slot's fetch is a
    /// debt (`crates/eidola-gui/STATE.md` → "Concurrency patterns"), which the
    /// unconditional re-list discharges. That re-list is also what bounds the
    /// caller's optimistic edit of the cached row: a refused write is
    /// reconciled back to what the database holds instead of leaving the
    /// Library, the window title, and the inspector's title field showing a
    /// name nothing ever persisted. The refusal itself wins `op_error`, while a
    /// re-list that fails resolves the *cell* (`Failed { prior }`, keeping the
    /// listing on screen) — [`crate::stores::settle_mutation`] owns that pairing.
    ///
    /// `op` returns the complete sentence to show, so each caller words its own
    /// refusal.
    fn write_then_relist<F>(&mut self, space_id: String, cx: &mut Context<Self>, op: F)
    where
        F: FnOnce(Arc<AppCore>) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
            + Send
            + 'static,
    {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.op_error = None;
        // Take over the read for the duration of the write.
        self.task = None;
        self.refresh_pending = false;
        let relist_core = core.clone();
        self.op_task = Some(cx.spawn(async move |this, cx| {
            let op_result = op(core).await;
            let list = bridge(relist_core, |c| async move { c.list_spaces(false).await }).await;
            let _ = this.update(cx, |this, cx| {
                if let Some(message) =
                    crate::stores::settle_mutation(&mut this.index, list, op_result)
                {
                    this.op_error = Some(SpaceOpError {
                        space_id: Some(space_id),
                        message,
                    });
                }
                this.op_task = None;
                if std::mem::take(&mut this.refresh_pending) {
                    this.refresh(cx);
                }
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Rename a space: update the cached row title immediately (optimistic, so
    /// the Library responds without a round trip), then write through
    /// `AppCore::rename_space` and re-list. The optimism lasts exactly as long
    /// as the round trip — a refused rename (a stale id, a local DB error)
    /// lands back on the stored title with its refusal in `op_error`, rather
    /// than leaving every surface reading the new name until an unrelated
    /// refresh silently takes it away. On stub stores (no backend) the
    /// optimistic update is the only visible effect — which is exactly what
    /// behavior tests assert.
    pub fn rename(&mut self, space_id: String, title: String, cx: &mut Context<Self>) {
        // Optimistic local update.
        if let Some(list) = self.index.value_mut()
            && let Some(row) = list.iter_mut().find(|s| s.id == space_id)
        {
            row.title = Some(title.clone());
        }
        cx.notify();

        let id = space_id.clone();
        self.write_then_relist(space_id, cx, move |core| {
            Box::pin(async move {
                bridge(
                    core,
                    move |c| async move { c.rename_space(id, title).await },
                )
                .await
                .map_err(|e| format!("Couldn't rename this space: {e}"))
            })
        });
    }

    /// Archive a space: drop the cached row immediately (so the Library
    /// responds without a backend round trip — and so stub tests exercise the
    /// local path), then archive core-side and re-list. Owning the write here
    /// is what makes it safe to start from a closing window: the core write
    /// completes regardless, and the bus reconciles every other window.
    pub fn archive(&mut self, space_id: String, cx: &mut Context<Self>) {
        // Optimistic local removal — edit whatever value is present, keeping
        // the cell's state (a stale or failed-with-prior listing stays what it
        // is; only its rows move).
        if let Some(list) = self.index.value_mut() {
            list.retain(|s| s.id != space_id);
        }
        cx.notify();

        let id = space_id.clone();
        self.write_then_relist(space_id, cx, move |core| {
            Box::pin(async move {
                match bridge(core, move |c| async move { c.archive_space(id).await }).await {
                    Ok(true) => Ok(()),
                    // The row was dropped on the assumption the write would
                    // take. `false` says it did not (the space was already
                    // archived, or is not there at all), so the removal was not
                    // this operation's to claim — say so, and let the re-list
                    // put the truth back.
                    Ok(false) => Err(
                        "Couldn't archive this space — it is no longer in your library."
                            .to_string(),
                    ),
                    Err(e) => Err(format!("Couldn't archive this space: {e}")),
                }
            })
        });
    }
}
