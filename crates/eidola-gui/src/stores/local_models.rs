//! `LocalModelsStore` — the local-inference domain: downloaded/downloading
//! models, running llama.cpp engines, and the resolved engine binary.
//! Refreshed at launch and on every `Change::LocalModels` (download
//! progress, engine lifecycle transitions — app-core emits them all).
//!
//! Operations (download/cancel/delete/load/unload) write through app-core;
//! the work itself is **core-owned** (transfer tasks and engine supervisors
//! live on the core tokio runtime, surviving any window), so the store's
//! per-operation task slots hold only the thin *initiating* call. Results
//! land via the bus → `refresh`; a failed initiation surfaces in
//! `op_error` (honest states — a button must never silently do nothing).

use std::collections::HashMap;
use std::sync::Arc;

use eidola_app_core::{AppCore, LocalModelInfo, LocalModelStatus, LocalModelsState};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct LocalModelsStore {
    app_core: Option<Arc<AppCore>>,
    state: Loadable<LocalModelsState>,
    /// Supersede slot for the snapshot refresh.
    task: Option<Task<()>>,
    /// Keyed per-model operation slots (download/cancel/delete/load/unload
    /// initiations) — independent per model id, replace-cancels per key.
    op_tasks: HashMap<String, Task<()>>,
    /// The most recent failed operation, as `(model_or_url, message)`.
    /// Cleared when any new operation begins.
    op_error: Option<String>,
}

impl LocalModelsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            state: Loadable::NotLoaded,
            task: None,
            op_tasks: HashMap::new(),
            op_error: None,
        }
    }

    /// A stub store with a fixture snapshot (tests).
    pub fn stub(state: Option<LocalModelsState>) -> Self {
        Self {
            app_core: None,
            state: match state {
                Some(s) => Loadable::loaded(s),
                None => Loadable::NotLoaded,
            },
            task: None,
            op_tasks: HashMap::new(),
            op_error: None,
        }
    }

    /// The current snapshot.
    pub fn state(&self) -> &Loadable<LocalModelsState> {
        &self.state
    }

    /// The local model list (empty unless loaded).
    pub fn models(&self) -> &[LocalModelInfo] {
        self.state
            .value()
            .map(|s| s.models.as_slice())
            .unwrap_or(&[])
    }

    /// Models currently loaded (ready to serve) — the set the model picker
    /// surfaces at the top of the dropdown.
    pub fn loaded_models(&self) -> Vec<LocalModelInfo> {
        self.models()
            .iter()
            .filter(|m| matches!(m.status, LocalModelStatus::Loaded { .. }))
            .cloned()
            .collect()
    }

    /// The resolved `llama-server` path, if one was found.
    pub fn engine_path(&self) -> Option<String> {
        self.state.value().and_then(|s| s.engine_path.clone())
    }

    /// The last failed operation's message, until the next operation.
    pub fn op_error(&self) -> Option<&str> {
        self.op_error.as_deref()
    }

    /// Refresh the snapshot. Fire-and-notify; the store owns the slot.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.state = std::mem::take(&mut self.state).to_loading();
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.local_models_state().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.state = std::mem::take(&mut this.state).resolve(result);
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Start a model download (curated entry or pasted URL). The transfer
    /// itself is core-owned; progress arrives via `Change::LocalModels`.
    pub fn download(&mut self, url: String, cx: &mut Context<Self>) {
        let key = format!("download:{url}");
        self.run_op(key, cx, move |c| async move {
            c.download_local_model(url).await.map(|_| ())
        });
    }

    /// Cancel an in-flight download.
    pub fn cancel_download(&mut self, id: String, cx: &mut Context<Self>) {
        let key = format!("cancel:{id}");
        self.run_op(key, cx, move |c| async move {
            c.cancel_local_model_download(id).await
        });
    }

    /// Delete a downloaded model from disk.
    pub fn delete(&mut self, id: String, cx: &mut Context<Self>) {
        let key = format!("delete:{id}");
        self.run_op(
            key,
            cx,
            move |c| async move { c.delete_local_model(id).await },
        );
    }

    /// Load a model (spawns its engine; resolves when ready or failed).
    /// Intermediate `Loading` state arrives via the bus.
    pub fn load(&mut self, id: String, cx: &mut Context<Self>) {
        let key = format!("load:{id}");
        self.run_op(
            key,
            cx,
            move |c| async move { c.load_local_model(id).await },
        );
    }

    /// Unload a model, stopping its engine.
    pub fn unload(&mut self, id: String, cx: &mut Context<Self>) {
        let key = format!("unload:{id}");
        self.run_op(
            key,
            cx,
            move |c| async move { c.unload_local_model(id).await },
        );
    }

    /// Shared operation shape: clear the standing error, run the initiating
    /// core call in this operation's keyed slot, and surface a failure in
    /// `op_error`. Success needs no local action — the bus-driven refresh
    /// carries the new state.
    fn run_op<F, Fut>(&mut self, key: String, cx: &mut Context<Self>, f: F)
    where
        F: FnOnce(Arc<AppCore>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), eidola_app_core::error::AppError>> + Send + 'static,
    {
        self.op_error = None;
        cx.notify();
        let Some(core) = self.app_core.clone() else {
            return;
        };
        let slot_key = key.clone();
        let task = cx.spawn(async move |this, cx| {
            let result = bridge(core, f).await;
            let _ = this.update(cx, |this, cx| {
                if let Err(e) = result {
                    this.op_error = Some(e.to_string());
                }
                this.op_tasks.remove(&slot_key);
                cx.notify();
            });
        });
        self.op_tasks.insert(key, task);
    }
}
