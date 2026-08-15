//! `SpacesStore` — the Library index (the list of non-archived spaces) **and**
//! the per-space entity registry.
//!
//! Per `crates/eidola-gui/STATE.md` ("Space entities — shared, registried"),
//! this store holds `HashMap<SpaceId, WeakEntity<Space>>`. Opening a space goes
//! through [`SpacesStore::open`], which gets-or-creates: two windows on the
//! same space share **one** `Space` entity (and one transcript load), which is
//! the structural fix for wave-2 bug 4 (a submit/stream in window A appears in
//! window B). [`SpacesStore::create`] is the ⌘N door: it mints the id, hands
//! back a registered entity in the same breath, and commits the row behind it
//! — so there is no window without a space, and no space outside the
//! registry.
//!
//! **The index's mutations follow the write-through rules** the other stores
//! do (`crates/eidola-gui/STATE.md` → "Concurrency patterns"): rename and
//! archive edit the cached row optimistically so the Library answers without a
//! round trip, then write in a slot of their **own** — never the refresh's,
//! because a rename emits the very `Change::SpaceIndex` that drives the refresh,
//! and a shared slot would let it cancel the write's own completion (the core
//! write still runs; the continuation carrying its refusal never does — a
//! refused write indistinguishable from a successful one). Every batch of
//! mutations ends in a read, so the optimism never outlives the round trip: a
//! refused write is reconciled back to what the database holds, with its refusal
//! in [`SpacesStore::op_error`].
//!
//! **The resolving read is taken after the last write, not carried from before
//! it** — see [`SpacesStore::write_then_reconcile`] — and each refused write
//! takes its own edit back through an inverse that stores no position: seats
//! come from the row's own sort key, names from its id, so no undo can go stale
//! behind a sibling's edit.
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

use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, SpaceInfo};
use gpui::{AppContext, Context, Entity, Task, WeakEntity};

use crate::bridge::bridge;
use crate::loadable::Loadable;
use crate::space::Space;
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
    /// The inverse of each in-flight mutation's optimistic edit, held here
    /// rather than only inside its task so that **a superseding write on the
    /// same space can take the dying one's edit back first**. Without that, the
    /// replaced op's undo dies with its cancelled task and the successor's undo
    /// captures *its optimistic string* as the value to restore — so a refused
    /// write whose re-list also failed could put back a name the database never
    /// held. Each settle takes its own entry out.
    op_undos: HashMap<String, UndoEdit>,
    /// A refresh signalled while any mutation held the read: deferred to the
    /// **last** one's completion, where it can only be fresher.
    refresh_pending: bool,
    /// The per-space entity registry. `WeakEntity` so a dropped window's space
    /// (with no other holder) is collected — `open` prunes dead weaks on miss.
    registry: HashMap<String, WeakEntity<Space>>,
    /// The in-flight inserts behind [`Self::create`], keyed by the space id
    /// each is committing. Owned here rather than by the entity or its window
    /// so a ⌘N window closed a keystroke later still leaves a whole space
    /// rather than a half-instantiated one — the store outlives every window
    /// (STATE.md, "owner = blast radius"). Each task removes its own key.
    instantiations: HashMap<String, Task<()>>,
    /// In-flight disposals of untouched spaces, keyed by the space each is
    /// about (see [`Self::window_closed`]). Store-owned for the same reason the
    /// inserts are: the window that triggered it is already gone.
    reap_tasks: HashMap<String, Task<()>>,
    /// Spaces whose last window closed while their own insert was still in
    /// flight. A space cannot be disposed of before it exists, so the close
    /// hands the id here and the insert's completion picks it up — which is
    /// what makes ⌘N-then-immediately-close leave nothing behind rather than
    /// waiting for the next launch's sweep.
    reap_after_creation: std::collections::HashSet<String>,
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
            op_undos: HashMap::new(),
            refresh_pending: false,
            registry: HashMap::new(),
            instantiations: HashMap::new(),
            reap_tasks: HashMap::new(),
            reap_after_creation: std::collections::HashSet::new(),
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
            op_undos: HashMap::new(),
            refresh_pending: false,
            registry: HashMap::new(),
            instantiations: HashMap::new(),
            reap_tasks: HashMap::new(),
            reap_after_creation: std::collections::HashSet::new(),
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
        if let Some(refusal) = refusal {
            self.record_op_error(space_id, refusal.to_string());
        }
        let list = listing.map_err(|e| eidola_app_core::error::AppError::Internal {
            message: e.to_string(),
        });
        self.index = std::mem::take(&mut self.index).resolve(list);
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

    /// Create a new space (⌘N) and hand back its registered entity **now**.
    ///
    /// The id is minted client-side ([`eidola_app_core::new_space_id`]) and the
    /// row — the default template instantiated, the same door `post` takes for
    /// a space-less save — commits behind it, so the window that opens on this
    /// entity is addressed by a real id from its first frame and the page stays
    /// instant. The registry holds it immediately, which is what makes every
    /// store-wide broadcast reach it by ordinary membership.
    ///
    /// A refused insert leaves no space, so it leaves no registry entry either:
    /// the entity is dropped from the registry and told
    /// ([`Space::creation_failed`]), which is what the window renders.
    pub fn create(&mut self, cx: &mut Context<Self>) -> Entity<Space> {
        let id = eidola_app_core::new_space_id();
        let app_core = self.app_core.clone();
        let entity = cx.new(|_| Space::created(app_core.clone(), id.clone()));
        self.registry.insert(id.clone(), entity.downgrade());

        let Some(core) = app_core else {
            return entity; // stub stores (tests): nothing to persist against
        };
        let key = id.clone();
        self.instantiations.insert(
            id.clone(),
            cx.spawn(async move |this, cx| {
                let result = bridge(core, move |c| async move {
                    c.create_space_with_id(id, None).await
                })
                .await;
                let _ = this.update(cx, |this, cx| {
                    this.instantiations.remove(&key);
                    match result {
                        // The index needs nothing here — the write emitted
                        // `Change::SpaceIndex` and the bus refreshes it the
                        // ordinary way. What the entity needs is the release:
                        // a mutation started before this commit is waiting on
                        // it (see `Space::creation_committed`).
                        Ok(_) => {
                            if let Some(space) =
                                this.registry.get(&key).and_then(WeakEntity::upgrade)
                            {
                                space.update(cx, |space, cx| space.creation_committed(cx));
                            }
                            // The window may already have closed over this
                            // insert (⌘N, then close a keystroke later). The
                            // space exists only now, so this is the earliest
                            // its disposal could be asked for.
                            if this.reap_after_creation.remove(&key) {
                                this.dispose_if_pristine(key.clone(), cx);
                            }
                        }
                        Err(e) => {
                            // Nothing was written, so there is nothing to
                            // dispose of — the refusal already drops the key.
                            this.reap_after_creation.remove(&key);
                            this.fail_creation(&key, e, cx);
                        }
                    }
                });
            }),
        );
        entity
    }

    /// The insert behind [`Self::create`] was refused: there is no space, so
    /// there is no registry entry either. Dropping the key is what keeps the
    /// registry a map of conversations that exist — a later `open` of that id
    /// would otherwise join an entity standing on an error. The entity itself
    /// is told, because its window is what has to say so — and because a save
    /// waiting on the insert has to be released with the refusal rather than
    /// left waiting on a settle that has already happened.
    fn fail_creation(&mut self, space_id: &str, error: AppError, cx: &mut Context<Self>) {
        let entity = self
            .registry
            .remove(space_id)
            .and_then(|weak| weak.upgrade());
        self.record_op_error(
            Some(space_id.to_string()),
            format!("Couldn't create the space: {error}"),
        );
        if let Some(space) = entity {
            space.update(cx, |space, cx| space.creation_failed(error, cx));
        }
        cx.notify();
    }

    /// Test seam: drive the production creation-failure arm. A local DB insert
    /// cannot be made to fail through the real seam (the same reason
    /// [`Self::settle_for_test`] exists), so this is how that quadrant is
    /// reachable from a test.
    #[doc(hidden)]
    pub fn fail_creation_for_test(
        &mut self,
        space_id: &str,
        error: AppError,
        cx: &mut Context<Self>,
    ) {
        self.fail_creation(space_id, error, cx);
    }

    /// A window drawing `space_id` has closed and no other window is drawing
    /// it — so **dispose of the space if nothing was ever done with it**.
    ///
    /// A space is created when its window opens, so an abandoned ⌘N (and every
    /// launch that opens a blank one) would otherwise leave a durable empty
    /// conversation in the Library forever. The GUI answers only "was that the
    /// last window" — the entity's own window list answers it
    /// ([`Space::has_other_open_window`]) — and every judgement about whether
    /// the space is *untouched* belongs to app-core, inside the transaction
    /// that deletes it.
    ///
    /// **A space whose insert is still in flight is deferred, not skipped.**
    /// The store owns the insert precisely so that a window closed a keystroke
    /// after ⌘N still leaves a whole space; disposing of one before it exists
    /// would answer "no such space" and leave the very orphan this exists to
    /// prevent, so the id waits for the insert's completion instead.
    pub fn window_closed(&mut self, space_id: String, cx: &mut Context<Self>) {
        if self.app_core.is_none() {
            return; // stub stores (tests): nothing durable to dispose of
        }
        if self.instantiations.contains_key(&space_id) {
            self.reap_after_creation.insert(space_id);
            return;
        }
        self.dispose_if_pristine(space_id, cx);
    }

    /// Ask app-core to delete `space_id` if it is provably pristine.
    ///
    /// **Whether a window took the space back is asked in the task, not here.**
    /// The entity outlives the frame its last window closes in, so a check at
    /// the close would see it either way; by the task's first poll the release
    /// has run, and a registry entry that still upgrades therefore means a
    /// *new* window is drawing this space. Asking too early only ever keeps a
    /// space, which is the direction to be wrong in.
    ///
    /// Residual, stated: a reopen landing after the call is issued races the
    /// delete and can lose it, leaving a window on a space that no longer
    /// exists. It costs no work — a pristine space contains none by definition
    /// — and closing it properly would mean app-core knowing what a window is.
    ///
    /// A refused or failed disposal says nothing to anyone: nobody asked for
    /// it, the Library is unchanged either way, and the next launch's sweep
    /// gets another go.
    fn dispose_if_pristine(&mut self, space_id: String, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        let key = space_id.clone();
        self.reap_tasks.insert(
            key.clone(),
            cx.spawn(async move |this, cx| {
                let reopened = this
                    .update(cx, |this, _| {
                        this.registry
                            .get(&key)
                            .and_then(WeakEntity::upgrade)
                            .is_some()
                    })
                    .unwrap_or(true);
                if reopened {
                    let _ = this.update(cx, |this, _| this.reap_tasks.remove(&key));
                    return;
                }
                let discarded = bridge(core, move |c| async move {
                    c.discard_if_pristine(space_id).await
                })
                .await;
                let _ = this.update(cx, |this, _| {
                    this.reap_tasks.remove(&key);
                    // The listing refreshes through the bus like any other
                    // write; what the store has to drop is the registry entry,
                    // so it stays a map of conversations that exist.
                    if matches!(discarded, Ok(true)) {
                        this.registry.remove(&key);
                    }
                });
            }),
        );
    }

    /// **Every live `Space` this store knows of** — the registry, minus the
    /// entries whose entity has been collected.
    fn live_spaces(&self) -> impl Iterator<Item = Entity<Space>> + '_ {
        self.registry.values().filter_map(WeakEntity::upgrade)
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
        for entity in self.live_spaces() {
            entity.update(cx, |space, cx| space.invalidate_incoming_references(cx));
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

    /// React to a `Change::Participants` by telling **every** live registered
    /// `Space` to re-read its transcript. Routed here from
    /// `stores::dispatch_change`.
    ///
    /// A transcript is where an author's *name* lives: `get_space_tree` resolves
    /// every post's byline and every reference edge's carried author identity
    /// through `COALESCE(space_participant.override_label, participant.label)`
    /// at read time, and nothing re-derives them afterwards. So a rename — which
    /// emits `Change::Participants` and nothing else — left every loaded window
    /// showing the old name in its post gutters, its minimap labels and its
    /// footnote rails, until an unrelated `Change::Space` or a reopen (Codex
    /// review, PR #292). The signal carries no id (a global agent is renamed
    /// everywhere at once, and an override is written per space), so the answer
    /// is every live space rather than one.
    ///
    /// **Not routed through `notify_space_changed`**, whose other two effects
    /// are about *posts*: nothing a participant edit does invalidates a cached
    /// incoming-reference index or a trace round. Each entity's own
    /// `is_busy` gate still applies, and it **defers rather than discards**.
    ///
    /// A space created a moment ago is an ordinary registry member — its row
    /// commits when its window opens — so a rename landing while its first
    /// read is in flight reaches it like any other (Codex review, PR #292).
    pub fn notify_participants_changed(&mut self, cx: &mut Context<Self>) {
        for entity in self.live_spaces() {
            entity.update(cx, |space, cx| space.invalidate_transcript(cx));
        }
    }

    /// The `Lagged` response for the live spaces: the bus dropped changes we
    /// can no longer name, so every open conversation drops everything it
    /// resolved at read time — its transcript (post gutters, minimap labels and
    /// the footnote rails' carried author identities), its incoming-reference
    /// index, and its trace rounds.
    ///
    /// **A store refresh does not reach any of it** (Codex review, PR #292).
    /// `stores::refresh_everything` re-read `SpacesStore::refresh` — the
    /// *Library index* — and nothing told the live `Space` entities anything,
    /// so a lag that swallowed a rename left every already-open transcript
    /// naming its authors as it did when it loaded, indefinitely, while the
    /// recovery claimed to have refreshed all state. The other two caches are
    /// the same gap in the same function: they are per-entity state whose only
    /// invalidation route is [`Self::notify_space_changed`], which the lagged
    /// path never called either.
    ///
    /// This is the whole of what `Change::Participants` + `Change::Space` do to
    /// a live space, minus the id — which is right, since a lag knows nothing
    /// about which space changed. Each entity's own busy gate still applies,
    /// and now defers rather than discards.
    pub fn notify_lagged(&mut self, cx: &mut Context<Self>) {
        for entity in self.live_spaces() {
            entity.update(cx, |space, cx| {
                space.invalidate_incoming_references(cx);
                space.invalidate_traces(cx);
                space.invalidate_transcript(cx);
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

    /// Dismiss the standing refusal about `space_id` — the per-space surface's
    /// ×, the counterpart of the Library's. A refusal is otherwise cleared only
    /// by the next write to the same space, so an unacknowledged one stands
    /// indefinitely; that is what let one band shadow another.
    pub fn dismiss_op_error_for(&mut self, space_id: &str, cx: &mut Context<Self>) {
        let before = self.op_errors.len();
        self.clear_op_error_for(Some(space_id));
        if self.op_errors.len() != before {
            cx.notify();
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
        // The space commits durably either way; only its window is abandoned
        // if the app retires meanwhile (`lifecycle::OpenIntent`).
        let intent = crate::lifecycle::intend_to_open(cx);
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
                            if intent.still_wanted(cx) {
                                crate::open_space_window(cx, stores.clone(), space.id);
                            }
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
    /// [`Self::write_then_reconcile`]), and a read that resolves after their own
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

    /// The write-through shape for an index mutation: apply the optimistic edit,
    /// run `op`, settle it — **in `space_id`'s own slot**, so a mutation on
    /// another space neither cancels this one nor is cancelled by it — and, once
    /// no write is left in flight, take the read that resolves the shared index.
    ///
    /// **The mutation takes over the read.** It drops any in-flight refresh,
    /// which may have been issued *before* this write and would re-stale the
    /// snapshot by resolving after it; cancelling another slot's fetch is a debt
    /// (`crates/eidola-gui/STATE.md` → "Concurrency patterns"), which the
    /// batch-end read discharges. A deferred refresh is dropped for the same
    /// reason and on the same promise: the read below runs strictly later, so it
    /// is everything the dropped refresh would have fetched.
    ///
    /// **The resolving read is taken after the last write, not carried from
    /// before it.** The cell is one listing for every space, so *which* read
    /// lands matters: a read each operation issues for itself begins right after
    /// *its own* write, which can be long before a sibling's commits — and if
    /// that operation is the one still standing when the last slot clears, it
    /// resolves the index with a snapshot that predates an accepted write (worse,
    /// a failed read then preserves that snapshot as `Failed { prior }`, so a
    /// durable rename or archive disappears behind a mere refresh error). Issuing
    /// the read only once `op_tasks` is empty makes "after every write of the
    /// batch" a property of *when it is taken*, which no interleaving can
    /// falsify. The read goes through [`Self::refresh`], so the rule re-arms
    /// recursively: a mutation starting while it is in flight drops it and owes
    /// the next one. Bounded and honest under continuous mutation — the newest
    /// batch-end read is the one that wins, and until it lands the cell shows the
    /// writes' own edits.
    ///
    /// That read is also what bounds `optimistic`, the caller's edit of the
    /// cached row: a refused write is reconciled back to what the database holds
    /// instead of leaving the Library, the window title, and the inspector's
    /// title field showing a name nothing ever persisted.
    ///
    /// **The edit is applied here, not by the caller**, so the undo that pairs
    /// with it is built in the same breath — the ordering the honest failure
    /// exit depends on cannot be forgotten at a call site. (`SpaceSettingsStore`
    /// takes its `advance` closure for the same reason.) `optimistic` returns
    /// **its own inverse**, which is what lets a refused write take back its edit
    /// without touching a sibling's — and what keeps the cell honest in the
    /// quadrant where the resolving read fails too, since `Failed { prior }` then
    /// keeps exactly the rows the undos have shaped. See [`settle_index_mutation`].
    ///
    /// `op` returns the complete sentence to show, so each caller words its own
    /// refusal.
    fn write_then_reconcile<F>(
        &mut self,
        space_id: String,
        cx: &mut Context<Self>,
        optimistic: impl FnOnce(&mut Vec<SpaceInfo>) -> UndoEdit,
        op: F,
    ) where
        F: FnOnce(Arc<AppCore>) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
            + Send
            + 'static,
    {
        // A write we are about to supersede leaves its optimism behind (its
        // undo dies with its cancelled task), and ours would then treat that
        // string as the value to restore. Take it back first, so the edit below
        // stands on what the database last told us.
        if let Some(superseded) = self.op_undos.remove(&space_id)
            && let Some(rows) = self.index.value_mut()
        {
            superseded(rows);
        }
        // The edit and its inverse, built together and with nothing between
        // them: this is what a refused write takes back.
        let undo = self
            .index
            .value_mut()
            .map(optimistic)
            .unwrap_or_else(|| -> UndoEdit { Box::new(|_| {}) });
        cx.notify();

        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.clear_op_error_for(Some(&space_id));
        // Take over the read for the duration of the write.
        self.task = None;
        self.refresh_pending = false;
        let key = space_id.clone();
        self.op_undos.insert(space_id.clone(), undo);
        self.op_tasks.insert(
            space_id.clone(),
            cx.spawn(async move |this, cx| {
                let op_result = op(core).await;
                let _ = this.update(cx, |this, cx| {
                    // Drop this op's slot *first*: whether it is the last one in
                    // flight is what decides who owes the resolving read.
                    this.op_tasks.remove(&key);
                    // Ours, unless a superseding write already took it back.
                    let undo = this
                        .op_undos
                        .remove(&key)
                        .unwrap_or_else(|| -> UndoEdit { Box::new(|_| {}) });
                    if let Some(message) = settle_index_mutation(&mut this.index, undo, op_result) {
                        this.record_op_error(Some(space_id), message);
                    }
                    // **The resolving read is taken here, not carried here.**
                    // It is issued only once no write is in flight, so it is
                    // taken strictly after every write of the batch has
                    // committed. A read each operation started for itself could
                    // not say that: it begins right after *its own* write, which
                    // may be long before a sibling's commits, and if that
                    // operation happened to be the last to settle it would land
                    // a snapshot predating an accepted write (and a failed read
                    // would preserve it as `prior`). `refresh` also re-arms the
                    // deferral recursively — a mutation starting during this
                    // read drops it and owes the next one — so the batch always
                    // ends on the newest read, and never on an older one.
                    if this.op_tasks.is_empty() {
                        this.refresh_pending = false;
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
        self.write_then_reconcile(
            space_id,
            cx,
            move |rows| {
                let mut prior: Option<Option<String>> = None;
                if let Some(row) = rows.iter_mut().find(|s| s.id == row_id) {
                    prior = Some(row.title.clone());
                    row.title = Some(optimistic_title);
                }
                // The inverse: this row's title, by id — the listing may have
                // been re-ordered by a sibling's re-list in the meantime.
                Box::new(move |rows: &mut Vec<SpaceInfo>| {
                    if let Some(prior) = prior
                        && let Some(row) = rows.iter_mut().find(|s| s.id == row_id)
                    {
                        row.title = prior;
                    }
                })
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
        self.write_then_reconcile(
            space_id,
            cx,
            // Optimistic local removal — an edit of whatever value is present,
            // so the cell's state is unchanged (a stale or failed-with-prior
            // listing stays what it is; only its rows move). Its inverse puts
            // the row back where the *listing's own order* puts it.
            //
            // **The position is derived, never stored.** A recorded slot is
            // stale the moment another optimistic removal shifts the list under
            // it: two archives of `[A, B, C]` record slot 0 apiece (B's after
            // A's removal), and undoing them in the order they settle can seat
            // them as `[B, A, C]` — an order the database never held, left
            // standing as `Failed { prior }` when the batch-end read fails too.
            // `list_spaces` orders by `last_activity_at DESC` and every row
            // carries that key, so the seat is a question the row answers about
            // itself, in any order, however many siblings moved meanwhile.
            move |rows| {
                let removed = rows
                    .iter()
                    .position(|s| s.id == row_id)
                    .map(|at| rows.remove(at));
                Box::new(move |rows: &mut Vec<SpaceInfo>| {
                    if let Some(row) = removed {
                        let at =
                            rows.partition_point(|r| r.last_activity_at >= row.last_activity_at);
                        rows.insert(at, row);
                    }
                })
            },
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

/// The inverse of one optimistic edit, built by the edit itself (see
/// [`SpacesStore::write_then_relist`]). It touches only the row its edit
/// touched, which is what lets one mutation take its edit back while a sibling's
/// is still pending on the same shared listing.
type UndoEdit = Box<dyn FnOnce(&mut Vec<SpaceInfo>)>;

/// Settle an index mutation: the write's outcome, the re-list's outcome, the
/// inverse of the edit this operation made, and whether this operation is the
/// last one in flight and may therefore resolve the shared cell.
///
/// [`crate::stores::settle_mutation`] resolves the cell and hands back the
/// refusal to record; the two things it cannot know are that *this* store's cell
/// was **edited in advance**, and that the cell is **shared with sibling
/// mutations** on other spaces.
///
/// **A refused write takes its edit back.** Normally the re-list does that — but
/// a re-list can fail too, and then `Failed { prior }` keeps the cell's current
/// rows on screen (the Library goes on showing them beside its retry, per
/// "Failed is not empty"). Those rows would be the optimistic edit: a title
/// `rename_space` refused, or a row a refused archive keeps hidden, standing
/// until some later read happens to succeed. Reconciliation-by-re-list is
/// exactly what is unavailable in that quadrant, so the undo keeps the promise
/// instead.
///
/// **The undo is this edit's inverse, not a snapshot of the whole listing.** A
/// snapshot is an absolute value and siblings compose by *delta*: restoring one
/// would erase a concurrent edit that has not settled yet, and a snapshot taken
/// after a sibling's edit would resurrect that sibling's refused edit when it
/// was restored later. An inverse touches one row and composes in any order. It
/// is not a re-application of optimism onto fetched truth — it runs only while
/// the cell still stands on the pre-fetch listing, and a fetched listing
/// replaces the cell wholesale with no undo outstanding.
///
/// **Order is load-bearing.** The undo lands *before* the resolve, so a
/// **successful** re-list — fresh truth, and strictly later than any edit —
/// supersedes it wholesale; the undo only ever shapes the `prior` that a failed
/// re-list keeps showing. And it is gated on the write being *refused*: a write
/// that landed durably is honest to keep on screen even when the read behind it
/// failed.
///
/// **Settling never resolves the cell.** This function carries no listing, by
/// construction: any listing an operation could hand it was read before the
/// batch finished writing, and resolving from one is precisely the defect the
/// batch-end read exists to prevent. The cell is resolved by the fresh read
/// [`SpacesStore::write_then_reconcile`] takes once no write is in flight — so
/// "the resolution reads the database *after* the last write" is a property of
/// when the read is issued, not of which captured snapshot gets applied. The
/// refusal is reported here either way: what an operation owes the reader is
/// never deferred.
fn settle_index_mutation(
    index: &mut Loadable<Vec<SpaceInfo>>,
    undo: UndoEdit,
    op: Result<(), String>,
) -> Option<String> {
    if op.is_err()
        && let Some(rows) = index.value_mut()
    {
        // Only where a value is present to be wrong: a cell holding no rows is
        // showing no optimistic edit either.
        undo(rows);
    }
    op.err()
}
#[cfg(test)]
mod tests {
    use super::{SpaceInfo, UndoEdit, settle_index_mutation};
    use crate::loadable::Loadable;
    use eidola_app_core::error::AppError;

    /// A listing row. `at` is `last_activity_at`, the key `list_spaces` orders
    /// by (descending) — the fixtures carry real ones because the archive undo
    /// derives its seat from them.
    fn row_at(id: &str, title: &str, at: i64) -> SpaceInfo {
        SpaceInfo {
            id: id.into(),
            title: Some(title.into()),
            snippet: None,
            created_at: at,
            last_activity_at: at,
            message_count: 0,
            archived_at: None,
        }
    }

    /// Rows whose order does not matter to the test.
    fn row(id: &str, title: &str) -> SpaceInfo {
        row_at(id, title, 0)
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

    /// Apply a rename to the cell exactly as `SpacesStore::rename` does, and hand
    /// back its inverse — the pair under test.
    fn rename(cell: &mut Loadable<Vec<SpaceInfo>>, id: &'static str, title: &str) -> UndoEdit {
        let title = title.to_string();
        let rows = cell.value_mut().expect("a cell with rows");
        let mut prior: Option<Option<String>> = None;
        if let Some(r) = rows.iter_mut().find(|s| s.id == id) {
            prior = Some(r.title.clone());
            r.title = Some(title);
        }
        Box::new(move |rows: &mut Vec<SpaceInfo>| {
            if let Some(prior) = prior
                && let Some(r) = rows.iter_mut().find(|s| s.id == id)
            {
                r.title = prior;
            }
        })
    }

    /// Likewise for archive — the seat derived from the row's own sort key.
    fn archive(cell: &mut Loadable<Vec<SpaceInfo>>, id: &'static str) -> UndoEdit {
        let rows = cell.value_mut().expect("a cell with rows");
        let removed = rows
            .iter()
            .position(|s| s.id == id)
            .map(|at| rows.remove(at));
        Box::new(move |rows: &mut Vec<SpaceInfo>| {
            if let Some(r) = removed {
                let at = rows.partition_point(|x| x.last_activity_at >= r.last_activity_at);
                rows.insert(at, r);
            }
        })
    }

    /// The batch-end read landing, as `refresh` applies it.
    fn resolving_read(
        cell: &mut Loadable<Vec<SpaceInfo>>,
        result: Result<Vec<SpaceInfo>, AppError>,
    ) {
        *cell = std::mem::take(cell).to_loading().resolve(result);
    }

    // -- The double-failure quadrant -----------------------------------------

    /// The write was refused *and* the read that would have taken the optimistic
    /// edit back failed too — so `Failed { prior }` keeps the cell's rows on
    /// screen (the Library shows them beside its retry), and those rows must not
    /// be the edit the database refused.
    #[test]
    fn a_refused_write_whose_read_also_fails_shows_the_pre_edit_rows() {
        let mut cell = Loadable::loaded(vec![row("s1", "Tides")]);
        let undo = rename(&mut cell, "s1", "Nile");
        let recorded = settle_index_mutation(
            &mut cell,
            undo,
            Err("Couldn't rename this space: space not found: s1".into()),
        );
        resolving_read(&mut cell, Err(read_err()));
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
    /// comes back, in its place, rather than staying hidden behind a failed read.
    #[test]
    fn a_refused_archive_whose_read_also_fails_puts_the_row_back() {
        let mut cell = Loadable::loaded(vec![
            row_at("s1", "Tides", 300),
            row_at("s2", "Bergamot", 200),
        ]);
        let undo = archive(&mut cell, "s1");
        settle_index_mutation(&mut cell, undo, Err("refused".into()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["Tides".to_string(), "Bergamot".to_string()],
            "the optimistic removal is undone when nothing else can undo it, \
             and the row lands where the listing's order puts it"
        );
    }

    /// **Two refused archives restore the database's order, in either settle
    /// order.** While the undo recorded a numeric slot, each archive of
    /// `[A, B, C]` recorded slot 0 — B's *after* A's removal had already shifted
    /// the list — so undoing them in one of the two orders seated them as
    /// `[B, A, C]`, an order the database never held, left standing as
    /// `Failed { prior }` when the batch-end read failed too. The seat is now
    /// derived from `last_activity_at`, the key `list_spaces` sorts by, so it
    /// cannot go stale behind a sibling's edit.
    #[test]
    fn two_refused_archives_restore_the_databases_order_first_settling_first() {
        let mut cell = Loadable::loaded(vec![
            row_at("a", "A", 300),
            row_at("b", "B", 200),
            row_at("c", "C", 100),
        ]);
        let undo_a = archive(&mut cell, "a");
        let undo_b = archive(&mut cell, "b");
        assert_eq!(titles(&cell), vec!["C".to_string()], "both rows are gone");

        settle_index_mutation(&mut cell, undo_a, Err("refused".into()));
        settle_index_mutation(&mut cell, undo_b, Err("refused".into()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            "the listing is the database's order, not the undos' arrival order"
        );
    }

    /// The same, settled the other way round — the order-independence is the
    /// property, so both orders are the test.
    #[test]
    fn two_refused_archives_restore_the_databases_order_last_settling_first() {
        let mut cell = Loadable::loaded(vec![
            row_at("a", "A", 300),
            row_at("b", "B", 200),
            row_at("c", "C", 100),
        ]);
        let undo_a = archive(&mut cell, "a");
        let undo_b = archive(&mut cell, "b");

        settle_index_mutation(&mut cell, undo_b, Err("refused".into()));
        settle_index_mutation(&mut cell, undo_a, Err("refused".into()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            "the listing is the database's order, not the undos' arrival order"
        );
    }

    /// An archive and a rename over the same listing: the rename's undo is
    /// by-id and moves nothing, so the archive's seat is unaffected by it —
    /// order-independence holds across *kinds* of edit, not just archives.
    #[test]
    fn a_refused_archive_and_a_refused_rename_both_land_honestly() {
        let mut cell = Loadable::loaded(vec![
            row_at("a", "A", 300),
            row_at("b", "B", 200),
            row_at("c", "C", 100),
        ]);
        let undo_a = archive(&mut cell, "a");
        let undo_b = rename(&mut cell, "b", "Bergamot");

        settle_index_mutation(&mut cell, undo_b, Err("refused".into()));
        settle_index_mutation(&mut cell, undo_a, Err("refused".into()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            "the row is back in its place and the refused name is gone"
        );
    }

    /// **Order matters.** The batch-end read is fresh truth and strictly later
    /// than every edit, so it supersedes them — including a listing that moved
    /// for reasons this operation knows nothing about.
    #[test]
    fn the_resolving_read_supersedes_the_undo() {
        let mut cell = Loadable::loaded(vec![row("s1", "Tides")]);
        let undo = rename(&mut cell, "s1", "Nile");
        settle_index_mutation(&mut cell, undo, Err("refused".into()));
        resolving_read(&mut cell, Ok(vec![row("s1", "Renamed in another window")]));
        assert_eq!(
            titles(&cell),
            vec!["Renamed in another window".to_string()],
            "the read wins over the undo, never the other way round"
        );
        assert!(!cell.is_loading() && cell.error().is_none());
    }

    /// The undo is gated on the *write* failing: a write that landed durably is
    /// honest to keep on screen even when the read behind it failed.
    #[test]
    fn an_accepted_write_whose_read_fails_keeps_its_edit() {
        let mut cell = Loadable::loaded(vec![row("s1", "Tides")]);
        let undo = rename(&mut cell, "s1", "Nile");
        let recorded = settle_index_mutation(&mut cell, undo, Ok(()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["Nile".to_string()],
            "the rename persisted — showing it is the truth, not optimism"
        );
        assert_eq!(recorded, None);
    }

    // -- Sibling mutations over the one shared cell --------------------------

    /// **A settling sibling contributes no listing at all** — the whole point of
    /// the batch-end read. Whatever a sibling read for itself was read before
    /// this still-running write committed, so a landed change would be dropped
    /// from view (and, if the resolving read then failed, kept out of
    /// `Failed { prior }` as well). Here both writes land and the read fails: the
    /// cell must still show both.
    #[test]
    fn a_landed_sibling_edit_survives_an_earlier_mutations_settle() {
        let mut cell = Loadable::loaded(vec![row("a", "A"), row("b", "B")]);
        let undo_a = rename(&mut cell, "a", "Anemone");
        let undo_b = rename(&mut cell, "b", "Bergamot");

        // A settles first, with B still in flight. It resolves nothing.
        settle_index_mutation(&mut cell, undo_a, Ok(()));
        assert_eq!(
            titles(&cell),
            vec!["Anemone".to_string(), "Bergamot".to_string()],
            "an earlier sibling's settle never replaces a pending edit"
        );

        // B settles last; the batch-end read is the only resolution, and it fails.
        settle_index_mutation(&mut cell, undo_b, Ok(()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["Anemone".to_string(), "Bergamot".to_string()],
            "both writes landed durably, so both belong in the preserved prior"
        );
        assert!(cell.error().is_some(), "with the read failure still shown");
    }

    /// A refused sibling takes back **only its own** edit. This is why the undo
    /// is an inverse and not a snapshot of the whole listing: a snapshot would
    /// erase the concurrent edit that has not settled yet.
    #[test]
    fn a_refused_sibling_takes_back_only_its_own_edit() {
        let mut cell = Loadable::loaded(vec![row("a", "A"), row("b", "B")]);
        let undo_a = rename(&mut cell, "a", "Anemone");
        let undo_b = rename(&mut cell, "b", "Bergamot");

        settle_index_mutation(&mut cell, undo_a, Err("refused".into()));
        assert_eq!(
            titles(&cell),
            vec!["A".to_string(), "Bergamot".to_string()],
            "A's refusal is undone; B's pending edit is untouched"
        );

        settle_index_mutation(&mut cell, undo_b, Ok(()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["A".to_string(), "Bergamot".to_string()],
            "and B's landed write survives the resolve"
        );
    }

    /// Two refusals in flight, settling in order: neither refused edit may be on
    /// screen at the end. A snapshot-based undo failed exactly here — the second
    /// op's snapshot was captured *after* the first op's edit, so restoring it
    /// resurrected an edit the database had already refused.
    #[test]
    fn two_refused_siblings_leave_neither_edit_behind() {
        let mut cell = Loadable::loaded(vec![row("a", "A"), row("b", "B")]);
        let undo_a = rename(&mut cell, "a", "Anemone");
        let undo_b = rename(&mut cell, "b", "Bergamot");

        settle_index_mutation(&mut cell, undo_a, Err("refused".into()));
        settle_index_mutation(&mut cell, undo_b, Err("refused".into()));
        resolving_read(&mut cell, Err(read_err()));
        assert_eq!(
            titles(&cell),
            vec!["A".to_string(), "B".to_string()],
            "no refused name survives, in either order"
        );
    }

    /// Settling reports the refusal whether or not this operation is the one that
    /// owes the resolving read: deferral is about the *index*, never about what
    /// an operation owes the reader.
    #[test]
    fn every_settle_reports_its_refusal() {
        let mut cell = Loadable::loaded(vec![row("s1", "Tides")]);
        let undo = rename(&mut cell, "s1", "Nile");
        let recorded = settle_index_mutation(&mut cell, undo, Err("refused".into()));
        assert_eq!(recorded.as_deref(), Some("refused"));
    }
}
