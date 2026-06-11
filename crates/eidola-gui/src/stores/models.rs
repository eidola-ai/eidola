//! `ModelsStore` — the model list + pricing, refreshed at launch and on
//! `Change::Config` (a base-URL flip points at a different upstream catalog).
//!
//! This is where the first chat window's model list comes from, fixing wave-2
//! bug 1 *structurally*: the refresh lives in its own supersede task slot with
//! no shared busy flag, so nothing else in flight (notably the startup wallet
//! recovery) can starve it.

use std::sync::Arc;

use eidola_app_core::{AppCore, ModelInfo};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct ModelsStore {
    app_core: Option<Arc<AppCore>>,
    models: Loadable<Vec<ModelInfo>>,
    /// Supersede slot: replacing it cancels the in-flight refresh, so the
    /// completing task is always the latest. `Loading` implies this is `Some`.
    task: Option<Task<()>>,
}

impl ModelsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            models: Loadable::NotLoaded,
            task: None,
        }
    }

    /// A stub store with a fixture model list (tests).
    pub fn stub(models: Vec<ModelInfo>) -> Self {
        Self {
            app_core: None,
            models: if models.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(models)
            },
            task: None,
        }
    }

    /// The current model-list snapshot.
    pub fn models(&self) -> &Loadable<Vec<ModelInfo>> {
        &self.models
    }

    /// The model list as a slice (empty unless `Loaded`/`Failed{prior}`).
    pub fn list(&self) -> &[ModelInfo] {
        self.models.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Refresh the model list. Fire-and-notify: the store owns the slot;
    /// callers observe the result. Re-fetch over data stays `Loaded{stale}`
    /// (no blank flash); an initial fetch shows `Loading`.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.models = std::mem::take(&mut self.models).to_loading();
        // Replacing the slot cancels any predecessor refresh.
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.available_models().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.models = std::mem::take(&mut this.models).resolve(result);
                this.task = None;
                cx.notify();
            });
        }));
        debug_assert!(
            self.task.is_some(),
            "Loading must imply a live task in the slot"
        );
        cx.notify();
    }
}
