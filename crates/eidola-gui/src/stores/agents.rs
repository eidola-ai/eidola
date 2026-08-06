//! `AgentsStore` — the shared **agent library** (Settings → Agents), task 36.
//!
//! Per `crates/eidola-gui/STATE.md`: one gpui entity owning the
//! `Loadable<Vec<GlobalAgentInfo>>` snapshot, a refresh slot plus **one write
//! slot per agent** (keyed, because editing one agent and retiring another are
//! independent — see [`AgentsStore::write_then_settle`]), its subscription to
//! `Change::Participants` (routed in `stores::dispatch_change`), and all
//! mutations of the global-agent domain.
//!
//! **Why not `ParticipantsStore`.** That store is keyed **per space** all the
//! way down — its snapshots, its refresh slots, its mutation slots and its
//! `op_error` are `HashMap<SpaceId, _>`, and its bus response (`refresh_all`)
//! means "re-list every cached space". A global agent belongs to no space, so it
//! has no key: it would have to arrive either under a sentinel key (a lie the
//! op-error map would then report under) or as a second, unkeyed set of fields
//! inside a store whose whole shape is the key. They are two domains that happen
//! to share one invalidation signal, which the bus dispatcher fans out to both
//! at no cost.
//!
//! The library's **edits** deliberately do *not* live here: editing a shared
//! agent is `AppCore::update_space_participant` — "edit everywhere" — the same
//! call the space inspector's Everyone mode makes, so both surfaces write the
//! same row through the same door. What is unique to the library is what only it
//! can do: **retire** an agent, and **open its notebook**.

use std::collections::HashMap;
use std::sync::Arc;

use eidola_app_core::{AppCore, GlobalAgentInfo, ParticipantUpdate};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct AgentsStore {
    app_core: Option<Arc<AppCore>>,
    agents: Loadable<Vec<GlobalAgentInfo>>,
    /// Supersede slot for the **list refresh** (launch, the pane's Retry, and
    /// every bus-driven `Change::Participants`).
    refresh_task: Option<Task<()>>,
    /// **One write slot per agent.** Editing one agent and retiring another are
    /// independent operations, and the roster offers both verbs on every row at
    /// once — so a store-wide slot would let the second write replace the first,
    /// dropping an in-flight write's continuation (its refusal, and the read
    /// that resolves the roster) or cancelling an unpolled one outright. The
    /// user's other action then either disappears or, worse, reads as accepted
    /// (Codex review, PR #279; the doctrine is `crates/eidola-gui/AGENTS.md` →
    /// "Independent mutations get keyed slots"). Two writes on the **same**
    /// agent still replace-cancel: that is one control, and last-wins is the
    /// intended reading.
    op_tasks: HashMap<String, Task<()>>,
    /// The last write error **per agent** — keyed exactly as the slots are,
    /// because a per-row band reading a store-wide slot could not tell "no
    /// refusal" from "another agent's refusal replaced mine".
    op_errors: HashMap<String, String>,
}

impl AgentsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            agents: Loadable::NotLoaded,
            refresh_task: None,
            op_tasks: HashMap::new(),
            op_errors: HashMap::new(),
        }
    }

    /// A stub store with a fixture roster (tests).
    pub fn stub(agents: Option<Vec<GlobalAgentInfo>>) -> Self {
        Self {
            app_core: None,
            agents: match agents {
                Some(list) => Loadable::loaded(list),
                None => Loadable::NotLoaded,
            },
            refresh_task: None,
            op_tasks: HashMap::new(),
            op_errors: HashMap::new(),
        }
    }

    pub fn agents(&self) -> &Loadable<Vec<GlobalAgentInfo>> {
        &self.agents
    }

    pub fn list(&self) -> &[GlobalAgentInfo] {
        self.agents.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Test-only: force the roster into `Failed` (no prior) to exercise the
    /// failed-initial-load rendering.
    #[doc(hidden)]
    pub fn set_failed_for_test(&mut self, error: &str) {
        self.agents = Loadable::Failed {
            error: eidola_app_core::error::AppError::Config {
                message: error.to_string(),
            },
            prior: None,
        };
    }

    /// Test-only: stand a refusal against one agent, the way a refused write
    /// leaves one — the stub stores have no backend, so this is how a view test
    /// reaches the per-row band (`SpacesStore::settle_for_test`'s reason).
    #[doc(hidden)]
    pub fn set_op_error_for_test(&mut self, participant_id: &str, message: &str) {
        self.op_errors
            .insert(participant_id.to_string(), message.to_string());
    }

    /// The standing refusal for one agent's last write, if any.
    pub fn op_error(&self, participant_id: &str) -> Option<&str> {
        self.op_errors.get(participant_id).map(String::as_str)
    }

    /// Whether any refusal stands (the pane's cheap "is there a band anywhere"
    /// question — it renders each under its own row).
    pub fn has_op_error(&self) -> bool {
        !self.op_errors.is_empty()
    }

    pub fn clear_op_error(&mut self, participant_id: &str, cx: &mut Context<Self>) {
        if self.op_errors.remove(participant_id).is_some() {
            cx.notify();
        }
    }

    /// Drop the refusals of agents the roster no longer carries. A refusal about
    /// an agent that has since been retired (here or in another window) has no
    /// row left to render under, and the fact it reported is moot — the agent is
    /// gone. Called by the pane's own roster reconciliation, so nothing is
    /// dropped on a listing that has not answered.
    pub fn forget_op_errors_absent_from(&mut self, present: &[String]) {
        self.op_errors
            .retain(|id, _| present.iter().any(|p| p == id));
    }

    /// Refresh the roster — its own supersede slot, deliberately not the
    /// mutation's.
    ///
    /// `Change::Participants` is a signal this store does not raise *and also
    /// one its own writes raise before returning*, so a shared slot would let an
    /// unrelated refresh cancel the gpui half of an in-flight retirement: the
    /// core write completes (`bridge` drops the tokio `JoinHandle`), but the
    /// continuation carrying its `op_error` and its re-list never runs, and a
    /// refused write becomes indistinguishable from an accepted one. The two
    /// slots are then *ordered* rather than merely parallel: a mutation takes
    /// over the read, and a refresh signalled meanwhile is deferred to its
    /// completion, where it can only be fresher.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        // A refresh signalled while any write is in flight is **dropped, not
        // queued**: every write's completion issues the batch-end read
        // unconditionally once the last slot clears, and that read runs strictly
        // later than this one would have — so it fetches everything the dropped
        // refresh was going to. (`SpacesStore` keeps a `refresh_pending` flag
        // for this, but its batch end is unconditional too, so nothing reads it;
        // a bool nobody consults is state that can only go stale.)
        if !self.op_tasks.is_empty() {
            return;
        }
        self.agents = std::mem::take(&mut self.agents).to_loading();
        self.refresh_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.list_global_agents().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.agents = std::mem::take(&mut this.agents).resolve(result);
                this.refresh_task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Start a fetch if the roster has never been loaded (the pane calls this on
    /// open). A no-op once a snapshot — or a `Failed` cell — exists; the pane's
    /// Retry is the way back from the latter.
    pub fn ensure(&mut self, cx: &mut Context<Self>) {
        if matches!(self.agents, Loadable::NotLoaded) {
            self.refresh(cx);
        }
    }

    /// The write-through shape: run `op` in **that agent's own slot**, record
    /// its refusal under the same key, and — once no write is left in flight —
    /// take the read that resolves the roster.
    ///
    /// **The mutation takes over the read.** It drops any in-flight refresh,
    /// which may have been issued before this write and would re-stale the
    /// snapshot by resolving after it; cancelling another slot's fetch is a debt
    /// (`crates/eidola-gui/STATE.md` → "Concurrency patterns") that the
    /// batch-end read below discharges — including the `Loading` a cancelled
    /// initial load left behind, which `refresh` resolves either way.
    ///
    /// **The resolving read is taken after the last write, not carried from
    /// before it.** The cell is one listing for every agent, so a read each
    /// operation issued for itself would begin right after *its own* write —
    /// possibly long before a sibling's commits — and the operation that
    /// happened to settle last would resolve the roster with a snapshot
    /// predating an accepted write. Issuing it only once `op_tasks` is empty
    /// makes "after every write of the batch" a property of *when it is taken*.
    /// It goes through [`Self::refresh`], so the rule re-arms recursively: a
    /// mutation starting during that read drops it and owes the next one.
    ///
    /// **No optimism, so no inverse.** Unlike `SpacesStore`, nothing here edits
    /// the cached listing before the write returns — a retirement or a rename
    /// shows when the re-list says so. That is what keeps this the *simple*
    /// keyed shape: with no local edit outstanding there is nothing a refused
    /// write must take back, and the roster can never show a value the database
    /// refused.
    fn write_then_settle<F>(&mut self, participant_id: String, cx: &mut Context<Self>, op: F)
    where
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
        self.op_errors.remove(&participant_id);
        // Take over the read for the duration of the write.
        self.refresh_task = None;
        let key = participant_id.clone();
        self.op_tasks.insert(
            participant_id,
            cx.spawn(async move |this, cx| {
                let op_result = op(core).await;
                let _ = this.update(cx, |this, cx| {
                    // Drop this op's slot *first*: whether it is the last one in
                    // flight is what decides who owes the resolving read.
                    this.op_tasks.remove(&key);
                    if let Err(message) = op_result {
                        this.op_errors.insert(key.clone(), message);
                    }
                    if this.op_tasks.is_empty() {
                        this.refresh(cx);
                    }
                    cx.notify();
                });
            }),
        );
        cx.notify();
    }

    /// Edit a shared agent's own config — **"edit everywhere"**, the same write
    /// the space inspector's Everyone mode makes (`update_space_participant`
    /// takes a participant id and edits the row itself). A space that overrode a
    /// field keeps its override; the rest follow.
    pub fn update_agent(
        &mut self,
        participant_id: String,
        update: ParticipantUpdate,
        cx: &mut Context<Self>,
    ) {
        let key = participant_id.clone();
        self.write_then_settle(key, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    // The library edits a shared identity by definition; a row
                    // that is not one is not this pane's to write.
                    c.update_space_participant(
                        participant_id,
                        update,
                        eidola_app_core::ExpectedScope::Global,
                    )
                    .await
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }

    /// Retire a shared agent — the library soft-remove, which also archives its
    /// notebook (one core transaction). **Not a demotion**: the row, its id and
    /// its authorship survive; there is no un-retire here because app-core
    /// offers none.
    pub fn retire(&mut self, participant_id: String, cx: &mut Context<Self>) {
        let key = participant_id.clone();
        self.write_then_settle(key, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.retire_participant(participant_id).await
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }
}
