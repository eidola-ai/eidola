//! `AgentsStore` — the shared **agent library** (Settings → Agents), task 36.
//!
//! Per `crates/eidola-gui/STATE.md`: one gpui entity owning the
//! `Loadable<Vec<GlobalAgentInfo>>` snapshot, one task slot per operation (a
//! refresh slot and a mutation slot, ordered exactly as `TemplatesStore`'s are),
//! its subscription to `Change::Participants` (routed in
//! `stores::dispatch_change`), and all mutations of the global-agent domain.
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
    /// Supersede slot for the **write-through ops** (each composes
    /// `write; re-list`). Separate from the refresh slot because a write emits
    /// `Change::Participants` itself — see [`AgentsStore::refresh`].
    op_task: Option<Task<()>>,
    /// A refresh signalled while a mutation held the read; run once it lands.
    refresh_pending: bool,
    /// The last write error, surfaced by the pane's op-error banner.
    op_error: Option<String>,
}

impl AgentsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            agents: Loadable::NotLoaded,
            refresh_task: None,
            op_task: None,
            refresh_pending: false,
            op_error: None,
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
            op_task: None,
            refresh_pending: false,
            op_error: None,
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

    pub fn op_error(&self) -> Option<&str> {
        self.op_error.as_deref()
    }

    pub fn clear_op_error(&mut self, cx: &mut Context<Self>) {
        if self.op_error.take().is_some() {
            cx.notify();
        }
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
        if self.op_task.is_some() {
            self.refresh_pending = true;
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

    /// The shared write-through shape: run `op`, then re-list. The re-list runs
    /// on **every** exit, failure included — it discharges the debt of having
    /// taken over the read, and `settle_mutation` resolves the cell either way
    /// (the write's refusal wins `op_error`; a re-list that itself fails lands
    /// `Failed { prior }` rather than a spinner with no live task).
    fn write_then_relist<F>(&mut self, cx: &mut Context<Self>, op: F)
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
        self.op_error = None;
        // Take over the read for the duration of the write.
        self.refresh_task = None;
        self.refresh_pending = false;
        let relist_core = core.clone();
        self.op_task = Some(cx.spawn(async move |this, cx| {
            let op_result = op(core).await;
            let list = bridge(relist_core, |c| async move { c.list_global_agents().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.op_error = crate::stores::settle_mutation(&mut this.agents, list, op_result);
                this.op_task = None;
                if std::mem::take(&mut this.refresh_pending) {
                    this.refresh(cx);
                }
                cx.notify();
            });
        }));
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
        self.write_then_relist(cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.update_space_participant(participant_id, update).await
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
        self.write_then_relist(cx, move |core| {
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
