//! `ParticipantsStore` — a space's participant membership (the per-space
//! Participants view's data source).
//!
//! Per `crates/eidola-gui/STATE.md`: one gpui entity keyed **per space** (a
//! `HashMap<SpaceId, Loadable<Vec<ParticipantInfo>>>` with one task slot per
//! space, mirroring `ModelsStore`'s per-backend catalogs), subscribed to
//! `Change::Participants` (routed in `stores::dispatch_change`, which refreshes
//! every cached space — the change carries no id), and owning *all* mutations
//! of the per-space participant domain. Each CRUD op composes `write; re-list`
//! in the space's slot so the store reconciles even on a bus-less test run.
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
    AppCore, NewParticipant, ParticipantInfo, ParticipantOverride, ParticipantUpdate,
};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct ParticipantsStore {
    app_core: Option<Arc<AppCore>>,
    /// One `Loadable` snapshot per opened space.
    spaces: HashMap<String, Loadable<Vec<ParticipantInfo>>>,
    /// One supersede task slot per space (list refresh + CRUD compose).
    tasks: HashMap<String, Task<()>>,
    /// The last write error, **keyed per space** — snapshots and task slots are
    /// space-keyed, so a store-wide error would cross-contaminate two open
    /// Participants windows (one space's failure banner appearing under another,
    /// and either op start clearing the other's). Each view reads only its own.
    op_errors: HashMap<String, String>,
}

const NOT_LOADED: Loadable<Vec<ParticipantInfo>> = Loadable::NotLoaded;

impl ParticipantsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            spaces: HashMap::new(),
            tasks: HashMap::new(),
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
            tasks: HashMap::new(),
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

    /// The last write error for `space_id`, if any (per-space, not store-wide).
    pub fn op_error(&self, space_id: &str) -> Option<&str> {
        self.op_errors.get(space_id).map(String::as_str)
    }

    pub fn clear_op_error(&mut self, space_id: &str, cx: &mut Context<Self>) {
        if self.op_errors.remove(space_id).is_some() {
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

    /// Refresh one space's participant list.
    pub fn refresh(&mut self, space_id: String, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        let entry = self
            .spaces
            .entry(space_id.clone())
            .or_insert(Loadable::NotLoaded);
        *entry = std::mem::take(entry).to_loading();
        let key = space_id.clone();
        self.tasks.insert(
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
                    this.tasks.remove(&key);
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

    /// The shared write-through shape: run `op`, then re-list `space_id`.
    fn write_then_relist<F>(&mut self, space_id: String, cx: &mut Context<Self>, op: F)
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
        self.op_errors.remove(&space_id);
        let relist_core = core.clone();
        let key = space_id.clone();
        self.tasks.insert(
            space_id.clone(),
            cx.spawn(async move |this, cx| {
                let op_result = op(core).await;
                let list = match &op_result {
                    Ok(()) => {
                        let s = space_id.clone();
                        Some(
                            bridge(relist_core, move |c| async move {
                                c.list_space_participants(s).await
                            })
                            .await,
                        )
                    }
                    Err(_) => None,
                };
                let _ = this.update(cx, |this, cx| {
                    match op_result {
                        Ok(()) => {
                            if let Some(Ok(list)) = list {
                                this.spaces.insert(key.clone(), Loadable::loaded(list));
                            }
                        }
                        Err(e) => {
                            this.op_errors.insert(key.clone(), e);
                        }
                    }
                    this.tasks.remove(&key);
                    cx.notify();
                });
            }),
        );
        cx.notify();
    }

    /// Add a new agent participant to a space.
    pub fn add(&mut self, space_id: String, participant: NewParticipant, cx: &mut Context<Self>) {
        let s = space_id.clone();
        self.write_then_relist(space_id, cx, move |core| {
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

    /// "Edit everywhere": write the participant's **own** config (edits the
    /// shared global everywhere, or the space-owned row for this space).
    pub fn update_everywhere(
        &mut self,
        space_id: String,
        participant_id: String,
        update: ParticipantUpdate,
        cx: &mut Context<Self>,
    ) {
        self.write_then_relist(space_id, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.update_space_participant(participant_id, update).await
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
        self.write_then_relist(space_id, cx, move |core| {
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

    /// Remove an agent participant from a space (the shared human can't be).
    pub fn remove(&mut self, space_id: String, participant_id: String, cx: &mut Context<Self>) {
        let s = space_id.clone();
        self.write_then_relist(space_id, cx, move |core| {
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
