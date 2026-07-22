//! `ModelsStore` — the per-backend model catalogs, refreshed at launch, on
//! `Change::Config` (a base-URL flip points at a different eidola catalog),
//! and on `Change::Backends` (the set of destinations changed).
//!
//! One [`BackendCatalog`] per enabled *fetch-based* backend (kinds `eidola`
//! and `openai` — the ones whose model lists come over HTTP). Engine-backed
//! kinds (`local`, `llamacpp`) don't fetch a catalog; their selectable
//! models are the loaded engines, which live in `LocalModelsStore`.
//!
//! Each catalog is its own `Loadable` with its own fetch slot, so one
//! backend's dead server never blanks another's list — the request panel
//! renders per-backend retry/refresh from exactly this state.
//!
//! This store still owns the fix for wave-2 bug 1 *structurally*: every
//! fetch lives in its own supersede task slot with no shared busy flag, so
//! nothing else in flight (notably the startup wallet recovery) can starve
//! the first window's model list.

use std::collections::HashMap;
use std::sync::Arc;

use eidola_app_core::{AppCore, BackendInfo, BackendKind, ModelInfo};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

/// One fetch-based backend's model catalog.
#[derive(Clone)]
pub struct BackendCatalog {
    pub backend: BackendInfo,
    pub models: Loadable<Vec<ModelInfo>>,
}

/// The empty cell handed out when a requested catalog doesn't exist.
static NOT_LOADED: Loadable<Vec<ModelInfo>> = Loadable::NotLoaded;

/// A stub `BackendInfo` for the eidola singleton (fixtures + the synchronous
/// placeholder before the first registry read lands).
pub fn eidola_backend_stub() -> BackendInfo {
    BackendInfo {
        id: eidola_app_core::EIDOLA_BACKEND_ID.into(),
        kind: BackendKind::Eidola,
        display_name: "Eidola".into(),
        enabled: true,
        base_url: None,
        has_api_key: false,
        models_dir: None,
        model_overrides: None,
        engine_path: None,
        auto_start: true,
        created_at: 0,
    }
}

pub struct ModelsStore {
    app_core: Option<Arc<AppCore>>,
    /// Per-backend catalogs in presentation order (eidola first, then
    /// external openai backends in creation order).
    catalogs: Vec<BackendCatalog>,
    /// Supersede slot for the registry read that drives a full refresh.
    list_task: Option<Task<()>>,
    /// Per-backend model-fetch slots, keyed by backend id —
    /// replace-cancels per key, so a retry supersedes its predecessor
    /// without touching the other backends' fetches.
    fetch_tasks: HashMap<String, Task<()>>,
}

impl ModelsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            catalogs: Vec::new(),
            list_task: None,
            fetch_tasks: HashMap::new(),
        }
    }

    /// A stub store with a fixture eidola model list (tests).
    pub fn stub(models: Vec<ModelInfo>) -> Self {
        let catalogs = if models.is_empty() {
            Vec::new()
        } else {
            vec![BackendCatalog {
                backend: eidola_backend_stub(),
                models: Loadable::loaded(models),
            }]
        };
        Self::stub_catalogs(catalogs)
    }

    /// A stub store with explicit per-backend catalogs (multi-backend
    /// scenes).
    pub fn stub_catalogs(catalogs: Vec<BackendCatalog>) -> Self {
        Self {
            app_core: None,
            catalogs,
            list_task: None,
            fetch_tasks: HashMap::new(),
        }
    }

    /// Every fetch-based backend's catalog, in presentation order.
    pub fn catalogs(&self) -> &[BackendCatalog] {
        &self.catalogs
    }

    /// One backend's catalog cell ([`Loadable::NotLoaded`] if absent).
    pub fn catalog(&self, backend_id: &str) -> &Loadable<Vec<ModelInfo>> {
        self.catalogs
            .iter()
            .find(|c| c.backend.id == backend_id)
            .map(|c| &c.models)
            .unwrap_or(&NOT_LOADED)
    }

    /// The eidola catalog's snapshot — the pre-backends compatibility view.
    pub fn models(&self) -> &Loadable<Vec<ModelInfo>> {
        self.catalog(eidola_app_core::EIDOLA_BACKEND_ID)
    }

    /// The eidola model list as a slice (empty unless loaded).
    pub fn list(&self) -> &[ModelInfo] {
        self.models().value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Test seam: force the eidola catalog into a failed state, modelling an
    /// offline/unreachable upstream. Lets the request-panel tests exercise
    /// the retry footer without a live (failing) backend.
    #[doc(hidden)]
    pub fn set_failed_for_test(&mut self, message: &str, cx: &mut Context<Self>) {
        let error = eidola_app_core::error::AppError::Internal {
            message: message.to_string(),
        };
        match self
            .catalogs
            .iter_mut()
            .find(|c| c.backend.id == eidola_app_core::EIDOLA_BACKEND_ID)
        {
            Some(cat) => {
                cat.models = Loadable::Failed {
                    error,
                    prior: cat.models.value().cloned(),
                };
            }
            None => self.catalogs.insert(
                0,
                BackendCatalog {
                    backend: eidola_backend_stub(),
                    models: Loadable::Failed { error, prior: None },
                },
            ),
        }
        cx.notify();
    }

    /// Full refresh: re-read the backend registry, reconcile the catalog
    /// set (dropping disabled/removed backends, keeping surviving
    /// backends' data visible as stale), and re-fetch every catalog.
    /// Fire-and-notify; the store owns every slot.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        // Mark what we have as in-flight immediately (no blank flash), and
        // ensure the eidola cell exists so the very first refresh renders an
        // honest `Loading` rather than nothing.
        if self.catalogs.is_empty() {
            self.catalogs.push(BackendCatalog {
                backend: eidola_backend_stub(),
                models: Loadable::NotLoaded,
            });
        }
        for cat in &mut self.catalogs {
            cat.models = std::mem::take(&mut cat.models).to_loading();
        }

        // Stage 1: the registry read. Its completion reconciles the catalog
        // set and spawns the per-backend fetches (stage 2).
        self.list_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.list_backends().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.list_task = None;
                match result {
                    Ok(backends) => this.reconcile_and_fetch(backends, cx),
                    Err(error) => {
                        // The registry itself failed (local DB error —
                        // rare). Surface it on every standing catalog.
                        for cat in &mut this.catalogs {
                            let prior = cat.models.value().cloned();
                            cat.models = Loadable::Failed {
                                error: error.clone(),
                                prior,
                            };
                        }
                    }
                }
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Re-fetch one backend's catalog (the panel's per-backend retry /
    /// refresh affordance). No-op for a backend without a standing catalog.
    pub fn refresh_backend(&mut self, backend_id: String, cx: &mut Context<Self>) {
        if self.app_core.is_none() {
            return;
        }
        let Some(cat) = self
            .catalogs
            .iter_mut()
            .find(|c| c.backend.id == backend_id)
        else {
            return;
        };
        cat.models = std::mem::take(&mut cat.models).to_loading();
        self.spawn_fetch(backend_id, cx);
        cx.notify();
    }

    /// Rebuild the catalog list from a fresh registry read, preserving each
    /// surviving backend's `Loadable` (already marked stale by `refresh`),
    /// then kick off every fetch.
    fn reconcile_and_fetch(&mut self, backends: Vec<BackendInfo>, cx: &mut Context<Self>) {
        let mut old: HashMap<String, Loadable<Vec<ModelInfo>>> = self
            .catalogs
            .drain(..)
            .map(|c| (c.backend.id.clone(), c.models))
            .collect();
        for backend in backends {
            if !backend.enabled
                || !matches!(backend.kind, BackendKind::Eidola | BackendKind::OpenAi)
            {
                continue;
            }
            let models = old.remove(&backend.id).unwrap_or(Loadable::NotLoaded);
            let models = models.to_loading();
            let id = backend.id.clone();
            self.catalogs.push(BackendCatalog { backend, models });
            self.spawn_fetch(id, cx);
        }
        // Dropped backends' fetch slots die here too (removed from the map
        // via retain), cancelling any in-flight fetch for them.
        let live: Vec<String> = self.catalogs.iter().map(|c| c.backend.id.clone()).collect();
        self.fetch_tasks.retain(|id, _| live.contains(id));
    }

    /// Spawn (replace-cancels) the model fetch for one backend id.
    fn spawn_fetch(&mut self, backend_id: String, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        let slot_key = backend_id.clone();
        let task = cx.spawn(async move |this, cx| {
            let id = backend_id.clone();
            let result = bridge(core, move |c| async move { c.backend_models(id).await }).await;
            let _ = this.update(cx, |this, cx| {
                if let Some(cat) = this
                    .catalogs
                    .iter_mut()
                    .find(|c| c.backend.id == backend_id)
                {
                    cat.models = std::mem::take(&mut cat.models).resolve(result);
                }
                this.fetch_tasks.remove(&backend_id);
                cx.notify();
            });
        });
        self.fetch_tasks.insert(slot_key, task);
    }
}
