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
//!
//! **Those slots are keyed per space, and so are the refusals.** One shared
//! mutation slot has the same defect one slot away: a second rename replaces
//! the first, and two spaces renamed a keystroke apart (two Library rows, two
//! windows) lose the first's write or its report. Mutations on different spaces
//! are independent work — the `Space` entity's `turn_runners` shape — and each
//! keeps its own refusal, because the space inspector reads per space and a
//! refusal that another space overwrote is a title field snapping back with no
//! reason given.

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

/// How many standing refusals are kept (see [`SpacesStore::op_errors`]). One per
/// space with an unacknowledged refusal, and a refusal only lands on a write the
/// user asked for, so the realistic ceiling is a handful; the cap exists so the
/// list cannot grow without bound in the pathological case. The oldest yields —
/// the surfaces read the most recent (Library) and the per-space one (inspector),
/// which are the two an oldest-first drop keeps longest.
const MAX_STANDING_OP_ERRORS: usize = 8;

pub struct SpacesStore {
    app_core: Option<Arc<AppCore>>,
    /// The Library index (archived excluded), newest activity first.
    index: Loadable<Vec<SpaceInfo>>,
    /// Supersede slot for the index **refresh** — deliberately not the
    /// mutation's (see the module docs).
    task: Option<Task<()>>,
    /// One supersede slot for an index **mutation** (rename / archive) **per
    /// space**, each composing `write; re-list`.
    ///
    /// Keyed, not one shared slot (the `Space` entity's `turn_runners` shape):
    /// mutations on *different* spaces are independent work and must coexist —
    /// a shared slot let a second rename replace the first, dropping the gpui
    /// half of a write already in flight (its refusal and its re-list lost, the
    /// silent-loss class this whole shape exists to cure) or cancelling the
    /// first operation outright before it was ever polled. Two Library rows,
    /// two windows, or a Library row and an inspector are all one keystroke
    /// apart.
    ///
    /// Two mutations on the *same* space still **replace-cancel**, deliberately:
    /// a space's title is edited by one control at a time (a row's rename field,
    /// an inspector's title), so last-wins is the right reading of two rapid
    /// writes to one — and the residual is bounded, since the superseded write's
    /// core call still runs (`bridge` cancels nothing) and the winner's own
    /// re-list reconciles the index either way. Only the superseded write's
    /// *message* is lost, and it was about the same space the winner reports on.
    op_tasks: HashMap<String, Task<()>>,
    /// A refresh signalled while any mutation held the read: deferred to the
    /// **last** one's completion, where it can only be fresher.
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
    /// The standing refusals — a create, a rename, or an archive the store
    /// could not complete — most recent last, at most one per key (a space id,
    /// or `None` for a create).
    ///
    /// A list rather than one slot for the same reason the task slots are keyed:
    /// two spaces can be refused at once, and a per-space surface reads only its
    /// own. A single slot would let the second refusal erase the first, leaving
    /// the first space's inspector showing a title field that snapped back with
    /// no reason given — the dishonesty this store's shape exists to prevent.
    /// The Library, which is the window over the whole index, shows the most
    /// recent of them.
    op_errors: Vec<SpaceOpError>,
}

impl SpacesStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            index: Loadable::NotLoaded,
            task: None,
            op_tasks: HashMap::new(),
            refresh_pending: false,
            registry: HashMap::new(),
            pending_blanks: Vec::new(),
            create_ops: HashMap::new(),
            next_create_op: 0,
            op_errors: Vec::new(),
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
            op_tasks: HashMap::new(),
            refresh_pending: false,
            registry: HashMap::new(),
            pending_blanks: Vec::new(),
            create_ops: HashMap::new(),
            next_create_op: 0,
            op_errors: Vec::new(),
        }
    }

    /// Test seam: apply the settle a mutation's completion applies — the
    /// re-list's outcome, plus the write's refusal when it was refused —
    /// without a backend to fail against. It runs the *production*
    /// [`crate::stores::settle_mutation`], so a view test driven through it is
    /// exercising the reconciliation shape the real path takes.
    ///
    /// `listing` is a `Result` because the re-list is one of the two things
    /// that can fail here, and the *read's* failure is what puts the index cell
    /// in `Failed` — the state the Library has to render honestly. A local DB
    /// read cannot be made to fail through the real seam, so this is how that
    /// quadrant is reachable from a test.
    #[doc(hidden)]
    pub fn settle_for_test(
        &mut self,
        space_id: Option<String>,
        listing: Result<Vec<SpaceInfo>, &str>,
        refusal: Option<&str>,
        cx: &mut Context<Self>,
    ) {
        let op = refusal.map_or(Ok(()), |r| Err(r.to_string()));
        let list = listing.map_err(|e| eidola_app_core::error::AppError::Internal {
            message: e.to_string(),
        });
        if let Some(message) = settle_index_mutation(&mut self.index, None, list, op) {
            self.record_op_error(space_id, message);
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

    /// The most recent refusal, whatever it was about — the Library's banner,
    /// which is the window over the whole index.
    pub fn op_error(&self) -> Option<&str> {
        self.op_errors.last().map(|e| e.message.as_str())
    }

    /// The standing refusal **about `space_id`** — what a per-space surface (the
    /// space inspector) shows, so a rename refused in one space never surfaces
    /// under another space's title field, and never goes unreported because
    /// another space was refused after it.
    pub fn op_error_for(&self, space_id: &str) -> Option<&str> {
        self.op_errors
            .iter()
            .find(|e| e.space_id.as_deref() == Some(space_id))
            .map(|e| e.message.as_str())
    }

    /// Dismiss the refusal the Library is showing — **that one**, not all of
    /// them: each standing refusal is about a different space and is its own
    /// message to acknowledge. A second one, if any, takes the banner.
    pub fn clear_op_error(&mut self, cx: &mut Context<Self>) {
        if self.op_errors.pop().is_some() {
            cx.notify();
        }
    }

    /// Record a refusal, replacing any standing one for the same key (a fresh
    /// refusal about a space supersedes that space's stale one) and keeping the
    /// list recency-ordered and bounded.
    fn record_op_error(&mut self, space_id: Option<String>, message: String) {
        self.op_errors.retain(|e| e.space_id != space_id);
        self.op_errors.push(SpaceOpError { space_id, message });
        if self.op_errors.len() > MAX_STANDING_OP_ERRORS {
            self.op_errors.remove(0);
        }
    }

    /// Drop the standing refusal for a key, if any — a new operation on it is
    /// about to answer the same question.
    fn clear_op_error_for(&mut self, space_id: Option<&str>) {
        self.op_errors.retain(|e| e.space_id.as_deref() != space_id);
    }

    // -- New Space from Template ------------------------------------------

    /// Instantiate a **specific** template into a new space and open its window.
    /// The op is **owned** here in a keyed task slot (never `.detach()`ed) so it
    /// runs to completion regardless of any window, and each activation is
    /// independent (no supersede) so a committed space always gets its window —
    /// no stranding. A failure is surfaced in [`SpacesStore::op_error`] (the
    /// Library banner), not silently discarded — and only the *previous create's*
    /// refusal is cleared as this one leaves, never another space's standing one.
    /// `stores` is passed through only to open the resulting window
    /// (`open_space_window` needs the bundle); it is not retained.
    pub fn create_from_template(
        &mut self,
        template_id: String,
        stores: Stores,
        cx: &mut Context<Self>,
    ) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.clear_op_error_for(None);
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
                            this.record_op_error(None, format!("Couldn't create the space: {e}"));
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
    /// A refresh signalled while a mutation is in flight is **deferred** to the
    /// completion of the **last** one rather than started beside them: a
    /// mutation has taken over the read for its duration (see
    /// [`Self::write_then_relist`]), and a read that resolves after their own
    /// re-lists can only be fresher than one racing them.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        if !self.op_tasks.is_empty() {
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
    /// on **every** exit, failure included — **in `space_id`'s own slot**, so a
    /// mutation on another space neither cancels this one nor is cancelled by it.
    ///
    /// **The mutation takes over the read.** It drops any in-flight refresh,
    /// which may have been issued *before* this write and would re-stale the
    /// snapshot by resolving after it; cancelling another slot's fetch is a
    /// debt (`crates/eidola-gui/STATE.md` → "Concurrency patterns"), which the
    /// unconditional re-list discharges. A deferred refresh is dropped for the
    /// same reason and on the same promise: this operation's own re-list, which
    /// runs strictly later, is what the refresh would have fetched.
    ///
    /// That re-list is also what bounds `optimistic`, the caller's edit of the
    /// cached row: a refused write is reconciled back to what the database holds
    /// instead of leaving the Library, the window title, and the inspector's
    /// title field showing a name nothing ever persisted. (With two mutations in
    /// flight over one shared cell, the first to settle briefly re-lists the
    /// second's optimistic edit away — a flicker of the durable truth, corrected
    /// by the second's own re-list, never a state that outlives the round trip.)
    /// The refusal itself joins `op_errors`, while a re-list that fails resolves
    /// the *cell* (`Failed { prior }`, keeping the listing on screen) —
    /// [`settle_index_mutation`] owns that pairing, including the case where
    /// **both** halves fail and the re-list is therefore unavailable to take the
    /// edit back.
    ///
    /// **The edit is applied here, not by the caller**, so the snapshot it
    /// replaces is captured in the same breath — the ordering the honest failure
    /// exit depends on cannot be forgotten at a call site. (`SpaceSettingsStore`
    /// takes its `advance` closure for the same reason.)
    ///
    /// `op` returns the complete sentence to show, so each caller words its own
    /// refusal.
    fn write_then_relist<F>(
        &mut self,
        space_id: String,
        cx: &mut Context<Self>,
        optimistic: impl FnOnce(&mut Vec<SpaceInfo>),
        op: F,
    ) where
        F: FnOnce(Arc<AppCore>) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
            + Send
            + 'static,
    {
        // The snapshot and the edit, in that order and with nothing between
        // them: this is what a refused write restores.
        let pre_optimistic = self.index.value().cloned();
        if let Some(rows) = self.index.value_mut() {
            optimistic(rows);
        }
        cx.notify();

        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.clear_op_error_for(Some(&space_id));
        // Take over the read for the duration of the write.
        self.task = None;
        self.refresh_pending = false;
        let relist_core = core.clone();
        let key = space_id.clone();
        self.op_tasks.insert(
            space_id.clone(),
            cx.spawn(async move |this, cx| {
                let op_result = op(core).await;
                let list = bridge(relist_core, |c| async move { c.list_spaces(false).await }).await;
                let _ = this.update(cx, |this, cx| {
                    if let Some(message) =
                        settle_index_mutation(&mut this.index, pre_optimistic, list, op_result)
                    {
                        this.record_op_error(Some(space_id), message);
                    }
                    this.op_tasks.remove(&key);
                    // The deferred refresh waits for the *last* mutation: an
                    // earlier one's re-list is not necessarily later than a
                    // sibling write still in flight.
                    if this.op_tasks.is_empty() && std::mem::take(&mut this.refresh_pending) {
                        this.refresh(cx);
                    }
                    cx.notify();
                });
            }),
        );
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
        let id = space_id.clone();
        let optimistic_title = title.clone();
        let row_id = space_id.clone();
        self.write_then_relist(
            space_id,
            cx,
            move |rows| {
                if let Some(row) = rows.iter_mut().find(|s| s.id == row_id) {
                    row.title = Some(optimistic_title);
                }
            },
            move |core| {
                Box::pin(async move {
                    bridge(
                        core,
                        move |c| async move { c.rename_space(id, title).await },
                    )
                    .await
                    .map_err(|e| format!("Couldn't rename this space: {e}"))
                })
            },
        );
    }

    /// Archive a space: drop the cached row immediately (so the Library
    /// responds without a backend round trip — and so stub tests exercise the
    /// local path), then archive core-side and re-list. Owning the write here
    /// is what makes it safe to start from a closing window: the core write
    /// completes regardless, and the bus reconciles every other window.
    pub fn archive(&mut self, space_id: String, cx: &mut Context<Self>) {
        let id = space_id.clone();
        let row_id = space_id.clone();
        self.write_then_relist(
            space_id,
            cx,
            // Optimistic local removal — an edit of whatever value is present,
            // so the cell's state is unchanged (a stale or failed-with-prior
            // listing stays what it is; only its rows move).
            move |rows| rows.retain(|s| s.id != row_id),
            move |core| {
                Box::pin(async move {
                    match bridge(core, move |c| async move { c.archive_space(id).await }).await {
                        Ok(true) => Ok(()),
                        // The row was dropped on the assumption the write would
                        // take. `false` says it did not (the space was already
                        // archived, or is not there at all), so the removal was
                        // not this operation's to claim — say so, and let the
                        // re-list put the truth back.
                        Ok(false) => Err(
                            "Couldn't archive this space — it is no longer in your library."
                                .to_string(),
                        ),
                        Err(e) => Err(format!("Couldn't archive this space: {e}")),
                    }
                })
            },
        );
    }
}

/// Settle an index mutation: the write's outcome, the re-list's outcome, and the
/// listing the optimistic edit replaced.
///
/// [`crate::stores::settle_mutation`] resolves the cell and hands back the
/// refusal to record; the one thing it cannot know is that *this* store's cell
/// was **edited in advance**. When the write is refused, the re-list is what
/// takes that edit back — but a re-list can fail too, and then
/// `Failed { prior }` keeps the cell's current rows on screen (the Library goes
/// on showing them beside its retry, per "Failed is not empty"). Those rows
/// would be the optimistic edit: a title `rename_space` refused, or a row a
/// refused archive keeps hidden, standing until some later read happens to
/// succeed. Reconciliation-by-re-list is exactly what is unavailable in that
/// quadrant, so the snapshot is what keeps the promise instead.
///
/// **Order is load-bearing.** The restore lands *before* the resolve, so a
/// **successful** re-list — fresh truth, and strictly later than any snapshot —
/// supersedes it wholesale; the restore only ever fills the `prior` that a
/// failed re-list keeps showing. And it is gated on the write being *refused*:
/// a write that landed durably is honest to keep on screen even when the read
/// behind it failed.
fn settle_index_mutation(
    index: &mut Loadable<Vec<SpaceInfo>>,
    pre_optimistic: Option<Vec<SpaceInfo>>,
    list: Result<Vec<SpaceInfo>, eidola_app_core::error::AppError>,
    op: Result<(), String>,
) -> Option<String> {
    if op.is_err()
        && let Some(pre) = pre_optimistic
        && let Some(rows) = index.value_mut()
    {
        // Only where a value is present to be wrong: a cell holding no rows is
        // showing no optimistic edit either.
        *rows = pre;
    }
    crate::stores::settle_mutation(index, list, op)
}

#[cfg(test)]
mod tests {
    use super::{SpaceInfo, settle_index_mutation};
    use crate::loadable::Loadable;
    use eidola_app_core::error::AppError;

    fn row(id: &str, title: &str) -> SpaceInfo {
        SpaceInfo {
            id: id.into(),
            title: Some(title.into()),
            snippet: None,
            created_at: 0,
            last_activity_at: 0,
            message_count: 0,
            archived_at: None,
        }
    }

    fn titles(cell: &Loadable<Vec<SpaceInfo>>) -> Vec<String> {
        cell.value()
            .map(|rows| rows.iter().filter_map(|r| r.title.clone()).collect())
            .unwrap_or_default()
    }

    fn read_err() -> AppError {
        AppError::Internal {
            message: "db read failed".into(),
        }
    }

    /// The double-failure quadrant. The write was refused *and* the re-list that
    /// would have taken the optimistic edit back failed too — so `Failed { prior }`
    /// keeps the cell's rows on screen (the Library shows them beside its retry),
    /// and those rows must not be the edit the database refused.
    #[test]
    fn a_refused_write_whose_relist_also_fails_shows_the_pre_edit_rows() {
        // The cell as `rename` left it: optimistically renamed.
        let mut cell = Loadable::loaded(vec![row("s1", "Nile")]);
        let recorded = settle_index_mutation(
            &mut cell,
            Some(vec![row("s1", "Tides")]),
            Err(read_err()),
            Err("Couldn't rename this space: space not found: s1".into()),
        );
        assert_eq!(
            titles(&cell),
            vec!["Tides".to_string()],
            "a name the database refused must not stand in as the prior snapshot"
        );
        assert!(
            cell.error().is_some(),
            "the read's failure still resolves the cell"
        );
        assert!(
            recorded.is_some(),
            "and the write's refusal still gets reported"
        );
    }

    /// Archive's shape of the same quadrant: the row a refused archive removed
    /// comes back rather than staying hidden behind a failed read.
    #[test]
    fn a_refused_archive_whose_relist_also_fails_puts_the_row_back() {
        let mut cell = Loadable::loaded(vec![row("s2", "Bergamot")]);
        settle_index_mutation(
            &mut cell,
            Some(vec![row("s1", "Tides"), row("s2", "Bergamot")]),
            Err(read_err()),
            Err("Couldn't archive this space — it is no longer in your library.".into()),
        );
        assert_eq!(
            titles(&cell),
            vec!["Tides".to_string(), "Bergamot".to_string()],
            "the optimistic removal is undone when nothing else can undo it"
        );
    }

    /// **Order matters.** A successful re-list is fresh truth and strictly later
    /// than the snapshot, so it supersedes it — including a listing that moved
    /// for reasons this operation knows nothing about.
    #[test]
    fn a_successful_relist_supersedes_the_snapshot() {
        let mut cell = Loadable::loaded(vec![row("s1", "Nile")]);
        settle_index_mutation(
            &mut cell,
            Some(vec![row("s1", "Tides")]),
            Ok(vec![row("s1", "Renamed in another window")]),
            Err("refused".into()),
        );
        assert_eq!(
            titles(&cell),
            vec!["Renamed in another window".to_string()],
            "the re-list wins over the snapshot, never the other way round"
        );
        assert!(!cell.is_loading() && cell.error().is_none());
    }

    /// The restore is gated on the *write* failing: a write that landed durably
    /// is honest to keep on screen even when the read behind it failed.
    #[test]
    fn an_accepted_write_whose_relist_fails_keeps_its_edit() {
        let mut cell = Loadable::loaded(vec![row("s1", "Nile")]);
        let recorded = settle_index_mutation(
            &mut cell,
            Some(vec![row("s1", "Tides")]),
            Err(read_err()),
            Ok(()),
        );
        assert_eq!(
            titles(&cell),
            vec!["Nile".to_string()],
            "the rename persisted — showing it is the truth, not optimism"
        );
        assert_eq!(recorded, None);
    }
}
