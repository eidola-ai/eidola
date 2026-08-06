//! `ParticipantsStore` — a space's participant membership (the per-space
//! Participants view's data source).
//!
//! Per `crates/eidola-gui/STATE.md`: one gpui entity keyed **per space** (a
//! `HashMap<SpaceId, Loadable<Vec<ParticipantInfo>>>` keyed by space, mirroring
//! `ModelsStore`'s per-backend catalogs), subscribed to `Change::Participants`
//! (routed in `stores::dispatch_change`, which refreshes every cached space —
//! the change carries no id), and owning *all* mutations of the per-space
//! participant domain. Each CRUD op composes `write; re-list` so the store
//! reconciles even on a bus-less test run.
//!
//! **One slot per operation, per space**: refreshes and mutations have separate
//! keyed slots, because a write emits `Change::Participants` itself and the
//! `refresh_all` that drives would otherwise cancel the writing op's own
//! completion — losing its `op_error` and its re-list. See
//! [`ParticipantsStore::refresh`] for the ordering the split then restores.
//!
//! The edit-everywhere-vs-override-here fork lives here as two distinct
//! methods: [`ParticipantsStore::update_everywhere`] writes the (possibly
//! shared) participant's own config; [`ParticipantsStore::set_override`] writes
//! the per-membership override for a referenced global. The GUI chooses which
//! to call; app-core enforces the semantics (see `AppCore`'s
//! `update_space_participant` / `set_space_participant_override`).

use std::collections::HashMap;
use std::sync::Arc;

use eidola_app_core::{
    AppCore, ExpectedScope, NewParticipant, ParticipantInfo, ParticipantOverride, ParticipantUpdate,
};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct ParticipantsStore {
    app_core: Option<Arc<AppCore>>,
    /// One `Loadable` snapshot per opened space.
    spaces: HashMap<String, Loadable<Vec<ParticipantInfo>>>,
    /// One supersede **list-refresh** slot per space.
    refresh_tasks: HashMap<String, Task<()>>,
    /// One **write slot per (space, participant)**, separate from the refresh
    /// slots above. Keyed by both halves because the roster offers a verb on
    /// every row at once — share this agent, remove that one — so two mutations
    /// a moment apart are independent, and a slot keyed by space alone let the
    /// second replace the first: unpolled, the first write never ran; polled,
    /// its refusal and the read that resolves the roster were discarded (Codex
    /// review, PR #279; the doctrine is `crates/eidola-gui/AGENTS.md` →
    /// "Independent mutations get keyed slots"). Two writes on the **same** row
    /// are **chained**, not replaced — see [`Self::write_then_settle`] — with a
    /// generation beside each task so a settling op can tell whether it is
    /// still the current one.
    op_tasks: HashMap<(String, String), (u64, Task<()>)>,
    /// Monotonic op counter; a task settles only while its key still names it.
    next_op_gen: u64,
    /// The last write error per `(space, participant)` — keyed exactly as the
    /// slots are. Space-keying alone cross-contaminated two open windows; adding
    /// the participant is what lets two refusals in *one* space both stand,
    /// which the section's band lists (see [`Self::op_errors_for`]).
    op_errors: HashMap<(String, String), String>,
}

/// The slot key an **add** writes under. A creation has no participant id yet,
/// and the add form is a single control the submit closes, so last-wins is the
/// same deliberate residual two writes on one row take. The sentinel cannot
/// collide with a real id (participant ids are UUIDs).
const ADD_KEY: &str = "\u{0}add";

const NOT_LOADED: Loadable<Vec<ParticipantInfo>> = Loadable::NotLoaded;

impl ParticipantsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            spaces: HashMap::new(),
            refresh_tasks: HashMap::new(),
            op_tasks: HashMap::new(),
            next_op_gen: 0,
            op_errors: HashMap::new(),
        }
    }

    /// A stub store seeded with one space's fixture participant list (tests).
    pub fn stub(seed: Option<(String, Vec<ParticipantInfo>)>) -> Self {
        let mut spaces = HashMap::new();
        if let Some((space_id, list)) = seed {
            spaces.insert(space_id, Loadable::loaded(list));
        }
        Self {
            app_core: None,
            spaces,
            refresh_tasks: HashMap::new(),
            op_tasks: HashMap::new(),
            next_op_gen: 0,
            op_errors: HashMap::new(),
        }
    }

    /// The participants of `space_id` (`NotLoaded` if never opened).
    pub fn participants(&self, space_id: &str) -> &Loadable<Vec<ParticipantInfo>> {
        self.spaces.get(space_id).unwrap_or(&NOT_LOADED)
    }

    /// Test-only: force a space's cell into `Failed` (no prior) to exercise the
    /// failed-initial-load rendering.
    #[doc(hidden)]
    pub fn set_failed_for_test(&mut self, space_id: &str, error: &str) {
        self.spaces.insert(
            space_id.to_string(),
            Loadable::Failed {
                error: eidola_app_core::error::AppError::Config {
                    message: error.to_string(),
                },
                prior: None,
            },
        );
    }

    pub fn list(&self, space_id: &str) -> &[ParticipantInfo] {
        self.participants(space_id)
            .value()
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Every standing refusal in `space_id`, as `(participant_id, message)`,
    /// ordered by participant so the surface is stable frame to frame.
    ///
    /// **All of them, not the newest.** Two independent writes can both be
    /// refused, and the inspector renders **one band per space** by design (the
    /// panel is 320px, and a band under every disclosure row would shout) — so
    /// the band lists what stands, each line naming its subject, rather than the
    /// store picking one and dropping the other. That is the same conclusion
    /// `AgentsStore` reached with per-row bands; only the surface differs,
    /// because a compact roster of disclosures has no row to hang a band under.
    pub fn op_errors_for(&self, space_id: &str) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .op_errors
            .iter()
            .filter(|((space, _), _)| space == space_id)
            .map(|((_, pid), message)| (pid.clone(), message.clone()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Acknowledge every refusal standing in `space_id` — the band's one ×,
    /// matching the one band it dismisses. It never implies a write succeeded.
    pub fn clear_op_error(&mut self, space_id: &str, cx: &mut Context<Self>) {
        let before = self.op_errors.len();
        self.op_errors.retain(|(space, _), _| space != space_id);
        if self.op_errors.len() != before {
            cx.notify();
        }
    }

    /// Start a fetch for `space_id` if it has never been loaded (the view calls
    /// this on open). A no-op if a snapshot is already present.
    pub fn ensure(&mut self, space_id: String, cx: &mut Context<Self>) {
        if !self.spaces.contains_key(&space_id) {
            self.refresh(space_id, cx);
        }
    }

    /// Refresh one space's participant list — in its own slot, deliberately not
    /// the mutation's (see the module docs for what a shared slot cost).
    ///
    /// A refresh signalled while any of that space's writes is in flight is
    /// **dropped, not queued**: every write's completion issues the batch-end
    /// read unconditionally once the space's last slot clears, and that read
    /// runs strictly later than this one would have, so it fetches everything
    /// the dropped refresh was going to (`AgentsStore` states the same rule; a
    /// queue flag nobody consults is state that can only go stale).
    pub fn refresh(&mut self, space_id: String, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        if self.writes_in_flight(&space_id) {
            return;
        }
        let entry = self
            .spaces
            .entry(space_id.clone())
            .or_insert(Loadable::NotLoaded);
        *entry = std::mem::take(entry).to_loading();
        let key = space_id.clone();
        self.refresh_tasks.insert(
            space_id.clone(),
            cx.spawn(async move |this, cx| {
                let result = bridge(core, move |c| async move {
                    c.list_space_participants(space_id).await
                })
                .await;
                let _ = this.update(cx, |this, cx| {
                    let entry = this
                        .spaces
                        .entry(key.clone())
                        .or_insert(Loadable::NotLoaded);
                    *entry = std::mem::take(entry).resolve(result);
                    this.refresh_tasks.remove(&key);
                    cx.notify();
                });
            }),
        );
        cx.notify();
    }

    /// Refresh every cached space — the bus response to `Change::Participants`
    /// (the change carries no space id, so re-read them all).
    pub fn refresh_all(&mut self, cx: &mut Context<Self>) {
        let keys: Vec<String> = self.spaces.keys().cloned().collect();
        for key in keys {
            self.refresh(key, cx);
        }
    }

    /// Whether any write is in flight for `space_id`.
    fn writes_in_flight(&self, space_id: &str) -> bool {
        self.op_tasks.keys().any(|(space, _)| space == space_id)
    }

    /// The shared write-through shape: run `op` in **that row's own slot**,
    /// record its refusal under the same key, and — once no write is left in
    /// flight for the space — take the read that resolves its roster.
    ///
    /// **A mutation takes over that space's read.** It drops any in-flight
    /// refresh, which may have been issued *before* this write and would
    /// re-stale the snapshot by resolving after it. Cancelling another slot's
    /// fetch is a debt (`crates/eidola-gui/STATE.md` → "Concurrency patterns")
    /// that the batch-end read below discharges on **every** exit, failure
    /// included — a failed write changed nothing durably, so its read simply
    /// re-establishes what the cancelled refresh was fetching, and `Loadable`
    /// resolves the cell either way (`Failed { prior }` keeps rows on screen).
    ///
    /// **The resolving read is taken after the last write of the batch, not
    /// carried from before it.** The cell is one listing for every row in the
    /// space, so a read each operation issued for itself would begin right
    /// after *its own* write — possibly long before a sibling's commits — and
    /// whichever settled last would resolve the roster with a snapshot
    /// predating an accepted write. Issuing it only once the space's slots are
    /// empty makes "after every write of the batch" a property of *when it is
    /// taken*. It goes through [`Self::refresh`], so the rule re-arms
    /// recursively: a mutation starting during that read drops it and owes the
    /// next one.
    ///
    /// **No optimism, so no inverse** (the `AgentsStore` half of the keyed
    /// shape): nothing here edits the cached listing before the write returns,
    /// so a refused write has nothing to take back and the roster can never
    /// show a value the database refused.
    fn write_then_settle<F>(
        &mut self,
        space_id: String,
        participant_id: String,
        cx: &mut Context<Self>,
        op: F,
    ) where
        F: FnOnce(
                Arc<AppCore>,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<(), String>> + Send>,
            > + Send
            + 'static,
    {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        let key = (space_id.clone(), participant_id);
        self.op_errors.remove(&key);
        // Take over this space's read for the duration of the write.
        self.refresh_tasks.remove(&space_id);
        self.next_op_gen += 1;
        let generation = self.next_op_gen;
        // **Chained, not replaced** — the same rule `AgentsStore` states in
        // full: dropping the predecessor's `Task` cancels only its gpui half,
        // so owning it and awaiting it is what makes last-wins true by
        // sequencing (Codex review, PR #279).
        let previous = self.op_tasks.remove(&key).map(|(_, task)| task);
        self.op_tasks.insert(
            key.clone(),
            (
                generation,
                cx.spawn(async move |this, cx| {
                    if let Some(previous) = previous {
                        previous.await;
                    }
                    let op_result = op(core).await;
                    let _ = this.update(cx, |this, cx| {
                        // Only the current generation settles; a superseded op
                        // reports nothing and must not take the slot that is now
                        // its successor's.
                        if this.op_tasks.get(&key).map(|(g, _)| *g) != Some(generation) {
                            return;
                        }
                        this.op_tasks.remove(&key);
                        if let Err(message) = op_result {
                            this.op_errors.insert(key.clone(), message);
                        }
                        if !this.writes_in_flight(&space_id) {
                            this.refresh(space_id.clone(), cx);
                        }
                        cx.notify();
                    });
                }),
            ),
        );
        cx.notify();
    }

    /// Add a new agent participant to a space.
    pub fn add(&mut self, space_id: String, participant: NewParticipant, cx: &mut Context<Self>) {
        let s = space_id.clone();
        self.write_then_settle(space_id, ADD_KEY.to_string(), cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.add_space_participant(s, participant).await
                })
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
            })
        });
    }

    /// **The grant** (task 37): give an agent membership of this space as an
    /// **observer** — the read-only role the blocked-follow → grant → retry
    /// loop asks for.
    ///
    /// **One core call, and it is not this store's job to say which.** A
    /// space-owned agent has to be shared first and a shared one merely joins,
    /// but *which* the row is is a fact another window can change between the
    /// picker's listing and the reader's confirmation — and a store branching
    /// on that snapshot asked for a promotion of an already-global row, which
    /// app-core refuses, for a membership it would happily have added; where
    /// the competing promotion granted this very space, it reported failure
    /// about a state that already held (Codex review, PR #280). So the verb is
    /// decided at the write, inside one transaction
    /// (`AppCore::grant_space_membership`) — the same move that put the
    /// persona inside the promoting transaction (PR #279), applied to the
    /// choice of operation rather than to its arguments. The pair was never
    /// allowed to be two calls anyway: promotion is one-way, so a grant refused
    /// after a promotion committed leaves an irreversible change nobody asked
    /// for.
    pub fn grant_membership(
        &mut self,
        space_id: String,
        participant_id: String,
        cx: &mut Context<Self>,
    ) {
        let space = space_id.clone();
        let pid = participant_id.clone();
        self.write_then_settle(space_id, participant_id, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.grant_space_membership(space, pid, eidola_app_core::MembershipRole::Observer)
                        .await
                        .map(|_| ())
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }

    /// "Edit everywhere": write the participant's **own** config (edits the
    /// shared global everywhere, or the space-owned row for this space).
    /// `expected` is the shape the editor was seeded on. Save and Share are the
    /// same control on one row, so the second replaces the first's slot — but a
    /// replaced write's core call keeps running, and the two do not share a
    /// premise: a Save composed against the **owned** row would otherwise land
    /// on a row promotion had just made global, republishing the old persona to
    /// every space it joins. Carried into the write, the expired premise makes
    /// it strike nothing (Codex review, PR #279).
    pub fn update_everywhere(
        &mut self,
        space_id: String,
        participant_id: String,
        update: ParticipantUpdate,
        expected: ExpectedScope,
        cx: &mut Context<Self>,
    ) {
        let key = participant_id.clone();
        self.write_then_settle(space_id, key, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.update_space_participant(participant_id, update, expected)
                        .await
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }

    /// "Override here": write the per-membership override for a referenced
    /// global (this space only; the shared global is untouched).
    pub fn set_override(
        &mut self,
        space_id: String,
        participant_id: String,
        override_: ParticipantOverride,
        cx: &mut Context<Self>,
    ) {
        let s = space_id.clone();
        let key = participant_id.clone();
        self.write_then_settle(space_id, key, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.set_space_participant_override(s, participant_id, override_)
                        .await
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }

    /// **Share** a space-owned agent across spaces — task 36's in-place
    /// promotion (`scope: 'space' → 'global'` on the same row), optionally
    /// saving `update` first.
    ///
    /// It rides the same `write; re-list` shape as every other mutation here,
    /// and that is the whole GUI story: the participant keeps its id, the space
    /// keeps it as a member with NULL overrides, and the re-list comes back with
    /// `source == "referenced"` — so the row's **"shared"** tag and the editor's
    /// override fork appear by themselves, with nothing view-side to update.
    ///
    /// **The optional persona is what makes the promise true.** The affordance
    /// is pressed from inside the open editor, and its confirmation says the
    /// agent keeps this space's persona *exactly as it is* — which a reader
    /// reads against the fields still on screen behind it. So the visible values
    /// are what gets shared.
    ///
    /// **They travel into the one core call, not as a write before it.** An
    /// update-then-promote pair kept in a single `bridge` closure fixes the
    /// gpui-side hazard (no refresh lands between them to replace this
    /// mutation's slot) but not the durable one: two calls are two
    /// transactions, and two windows share one `AppCore`. Let another window
    /// share or remove the same agent in between and the persona commits — on a
    /// row that is now **global**, so in every space that follows it — while the
    /// promotion is refused and this store reports that sharing failed. One
    /// closure is not one transaction; `AppCore::promote_participant` applies
    /// the persona inside the promoting transaction, behind the same guard, so
    /// every refusal leaves zero trace (`crates/eidola-gui/AGENTS.md` →
    /// "Multi-call ops stay in one bridge closure"). Regression:
    /// `a_share_that_loses_the_race_writes_no_persona` (`tests/stores.rs`).
    ///
    /// **One-way**: app-core offers no demotion (it would strand memberships and
    /// memory), so there is no unshare here either — retirement is the
    /// soft-remove.
    pub fn promote(
        &mut self,
        space_id: String,
        participant_id: String,
        persona: Option<ParticipantUpdate>,
        cx: &mut Context<Self>,
    ) {
        let key = participant_id.clone();
        self.write_then_settle(space_id, key, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.promote_participant(participant_id, persona, None).await
                })
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
            })
        });
    }

    /// Remove an agent participant from a space (the shared human can't be).
    pub fn remove(&mut self, space_id: String, participant_id: String, cx: &mut Context<Self>) {
        let s = space_id.clone();
        let key = participant_id.clone();
        self.write_then_settle(space_id, key, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.remove_space_participant(s, participant_id).await
                })
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
            })
        });
    }
}
