//! `BackendsStore` — the backend registry: which inference destinations are
//! configured (the eidola and local singletons plus user-added
//! OpenAI-compatible / llama.cpp servers), refreshed at launch and on every
//! `Change::Backends`.
//!
//! Operations (add/enable/update/remove) write through app-core; results
//! land via the bus → `refresh`. A failed initiation surfaces in
//! `op_error` (honest states — a button must never silently do nothing).

use std::collections::HashMap;
use std::sync::Arc;

use eidola_app_core::{AppCore, BackendInfo, BackendUpdate, NewBackend};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct BackendsStore {
    app_core: Option<Arc<AppCore>>,
    backends: Loadable<Vec<BackendInfo>>,
    /// Supersede slot for the snapshot refresh.
    task: Option<Task<()>>,
    /// Keyed per-operation slots — independent per backend id,
    /// replace-cancels per key.
    op_tasks: HashMap<String, Task<()>>,
    /// The most recent failed operation's message. Cleared when any new
    /// operation begins.
    op_error: Option<String>,
}

impl BackendsStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            backends: Loadable::NotLoaded,
            task: None,
            op_tasks: HashMap::new(),
            op_error: None,
        }
    }

    /// A stub store with a fixture backend list (tests).
    pub fn stub(backends: Vec<BackendInfo>) -> Self {
        Self {
            app_core: None,
            backends: if backends.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(backends)
            },
            task: None,
            op_tasks: HashMap::new(),
            op_error: None,
        }
    }

    /// The current snapshot.
    pub fn state(&self) -> &Loadable<Vec<BackendInfo>> {
        &self.backends
    }

    /// The backend list as a slice (empty unless loaded).
    pub fn list(&self) -> &[BackendInfo] {
        self.backends.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// One backend by id, if present in the snapshot.
    pub fn get(&self, id: &str) -> Option<&BackendInfo> {
        self.list().iter().find(|b| b.id == id)
    }

    /// Whether a backend is present and enabled. Defaults to `true` for the
    /// singletons while the snapshot hasn't loaded — the optimistic answer
    /// keeps launch-time gating (onboarding auto-open) from flickering.
    pub fn is_enabled(&self, id: &str) -> bool {
        match self.get(id) {
            Some(b) => b.enabled,
            None => !self.backends.has_value(),
        }
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
        self.backends = std::mem::take(&mut self.backends).to_loading();
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.list_backends().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.backends = std::mem::take(&mut this.backends).resolve(result);
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Add an external backend. The bus-driven refresh carries the new row.
    pub fn add(&mut self, new: NewBackend, cx: &mut Context<Self>) {
        let key = format!("add:{}", new.id);
        self.run_op(key, cx, move |c| async move {
            c.add_backend(new).await.map(|_| ())
        });
    }

    /// Enable/disable a backend (singletons included).
    pub fn set_enabled(&mut self, id: String, enabled: bool, cx: &mut Context<Self>) {
        // Optimistic flip so the toggle answers immediately; the bus-driven
        // refresh reconciles.
        if let Some(list) = self.backends.value_mut()
            && let Some(b) = list.iter_mut().find(|b| b.id == id)
        {
            b.enabled = enabled;
        }
        let key = format!("enable:{id}");
        self.run_op(key, cx, move |c| async move {
            c.set_backend_enabled(id, enabled).await
        });
    }

    /// Update an external backend's configuration.
    pub fn update_config(&mut self, id: String, update: BackendUpdate, cx: &mut Context<Self>) {
        let key = format!("update:{id}");
        self.run_op(key, cx, move |c| async move {
            c.update_backend(id, update).await
        });
    }

    /// Soft-remove an external backend.
    pub fn remove(&mut self, id: String, cx: &mut Context<Self>) {
        // Optimistic removal from the cached list; the bus refresh confirms.
        if let Some(list) = self.backends.value_mut() {
            list.retain(|b| b.id != id);
        }
        let key = format!("remove:{id}");
        self.run_op(key, cx, move |c| async move { c.remove_backend(id).await });
    }

    /// Shared operation shape (mirrors `LocalModelsStore::run_op`): clear
    /// the standing error, run the initiating core call in this operation's
    /// keyed slot, and surface a failure in `op_error`. Success needs no
    /// local action — the bus-driven refresh carries the new state.
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
