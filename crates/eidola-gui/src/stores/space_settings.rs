//! `SpaceSettingsStore` — a space's own settings (cascade limit, router
//! model), the space inspector's data source.
//!
//! Per `crates/eidola-gui/STATE.md`: one gpui entity keyed **per space** (the
//! `ParticipantsStore` shape), refreshed on `Change::Space` — which carries the
//! id, so only that space re-reads — and owning all mutations of the domain.
//! The settings are durable per-space data, so two windows on one space must
//! agree: they observe this store, not view-local copies.
//!
//! **One slot per operation, per space.** A write here emits `Change::Space`
//! itself, and the refresh that drives would cancel the writing op's own
//! completion if they shared a slot — losing its `op_error` and its re-read.
//! The split is ordered exactly as [`crate::stores::participants`] documents:
//! a mutation takes over the read, a refresh signalled meanwhile defers to it,
//! and every exit re-reads (`crate::stores::settle_mutation` resolves the cell
//! either way).
//!
//! The space **title** is deliberately *not* here: it belongs to the Library
//! index (`SpacesStore`, which already owns rename), and the inspector's title
//! row writes through that store so the Library, the window title, and the
//! inspector can't disagree.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eidola_app_core::{AppCore, SpaceSettings};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct SpaceSettingsStore {
    app_core: Option<Arc<AppCore>>,
    /// One `Loadable` snapshot per opened space.
    spaces: HashMap<String, Loadable<SpaceSettings>>,
    /// One supersede **read** slot per space.
    refresh_tasks: HashMap<String, Task<()>>,
    /// One supersede **mutation** slot per space (each composes `write;
    /// re-read`), separate from the read slot above.
    op_tasks: HashMap<String, Task<()>>,
    /// Spaces whose refresh was signalled while their mutation held the read.
    refresh_pending: HashSet<String>,
    /// The last write error, keyed per space (two inspectors on two spaces
    /// must never show each other's refusal).
    op_errors: HashMap<String, String>,
}

const NOT_LOADED: Loadable<SpaceSettings> = Loadable::NotLoaded;

impl SpaceSettingsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            spaces: HashMap::new(),
            refresh_tasks: HashMap::new(),
            op_tasks: HashMap::new(),
            refresh_pending: HashSet::new(),
            op_errors: HashMap::new(),
        }
    }

    /// A stub store seeded with one space's fixture settings (tests).
    pub fn stub(seed: Option<(String, SpaceSettings)>) -> Self {
        let mut spaces = HashMap::new();
        if let Some((space_id, settings)) = seed {
            spaces.insert(space_id, Loadable::loaded(settings));
        }
        Self {
            app_core: None,
            spaces,
            refresh_tasks: HashMap::new(),
            op_tasks: HashMap::new(),
            refresh_pending: HashSet::new(),
            op_errors: HashMap::new(),
        }
    }

    /// The settings cell for `space_id` (`NotLoaded` if never opened).
    pub fn settings(&self, space_id: &str) -> &Loadable<SpaceSettings> {
        self.spaces.get(space_id).unwrap_or(&NOT_LOADED)
    }

    /// The last write error for `space_id`, if any.
    pub fn op_error(&self, space_id: &str) -> Option<&str> {
        self.op_errors.get(space_id).map(String::as_str)
    }

    pub fn clear_op_error(&mut self, space_id: &str, cx: &mut Context<Self>) {
        if self.op_errors.remove(space_id).is_some() {
            cx.notify();
        }
    }

    /// Test-only: force a space's cell into `Failed` (no prior), to exercise
    /// the failed-initial-load rendering.
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

    /// Start a read for `space_id` if it has never been loaded (the inspector
    /// calls this when it opens). A no-op once a snapshot is present.
    pub fn ensure(&mut self, space_id: String, cx: &mut Context<Self>) {
        if !self.spaces.contains_key(&space_id) {
            self.refresh(space_id, cx);
        }
    }

    /// Re-read one space's settings. Deferred when that space's mutation holds
    /// the read (a read resolving after the mutation's own re-read can only be
    /// fresher than one racing it).
    pub fn refresh(&mut self, space_id: String, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        if self.op_tasks.contains_key(&space_id) {
            self.refresh_pending.insert(space_id);
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
                let result = bridge(
                    core,
                    move |c| async move { c.space_settings(space_id).await },
                )
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

    /// Re-read a space **only if it is already cached** — the bus response to
    /// `Change::Space(id)`. A space nobody has an inspector open on stays
    /// untouched (every post emits `Change::Space`, and settings do not move
    /// with the transcript).
    pub fn refresh_if_cached(&mut self, space_id: &str, cx: &mut Context<Self>) {
        if self.spaces.contains_key(space_id) {
            self.refresh(space_id.to_string(), cx);
        }
    }

    /// Re-read every cached space (the `Lagged` response).
    pub fn refresh_all(&mut self, cx: &mut Context<Self>) {
        let keys: Vec<String> = self.spaces.keys().cloned().collect();
        for key in keys {
            self.refresh(key, cx);
        }
    }

    /// The write-through shape: run `op`, then re-read `space_id` — on **every**
    /// exit, failure included, because the mutation cancelled whatever refresh
    /// was in flight (STATE.md's cancellation debt). See
    /// [`crate::stores::participants::ParticipantsStore`] for the full
    /// rationale; this is the same machine over a one-row domain.
    fn write_then_reread<F>(&mut self, space_id: String, cx: &mut Context<Self>, op: F)
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
        self.refresh_tasks.remove(&space_id);
        self.refresh_pending.remove(&space_id);
        let read_core = core.clone();
        let key = space_id.clone();
        self.op_tasks.insert(
            space_id.clone(),
            cx.spawn(async move |this, cx| {
                let op_result = op(core).await;
                let read = bridge(read_core, move |c| async move {
                    c.space_settings(space_id).await
                })
                .await;
                let _ = this.update(cx, |this, cx| {
                    let cell = this
                        .spaces
                        .entry(key.clone())
                        .or_insert(Loadable::NotLoaded);
                    if let Some(e) = crate::stores::settle_mutation(cell, read, op_result) {
                        this.op_errors.insert(key.clone(), e);
                    }
                    this.op_tasks.remove(&key);
                    if this.refresh_pending.remove(&key) {
                        this.refresh(key.clone(), cx);
                    }
                    cx.notify();
                });
            }),
        );
        cx.notify();
    }

    /// Set the space's cascade limit (app-core enforces the floor).
    pub fn set_cascade_limit(&mut self, space_id: String, limit: i64, cx: &mut Context<Self>) {
        let s = space_id.clone();
        self.write_then_reread(space_id, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.set_space_cascade_limit(s, limit).await
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }

    /// Set (or clear, with `None` — **Off**) the space's may-decline router
    /// model. A remote reference bills an inference per post, which is why the
    /// picker states the cost inline; app-core validates the backend.
    pub fn set_router_model(
        &mut self,
        space_id: String,
        router_model: Option<String>,
        cx: &mut Context<Self>,
    ) {
        let s = space_id.clone();
        self.write_then_reread(space_id, cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.set_space_router_model(s, router_model).await
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }
}
