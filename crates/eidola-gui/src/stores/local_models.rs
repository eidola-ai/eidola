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

    /// Test seam: seat a snapshot on a **backend-backed** store, so a test can
    /// put a row on screen and still exercise the real op slots. (A stub store
    /// takes its fixture through `LocalModelsStore::stub`, but that one starts
    /// no tasks, which is exactly what a pending-operation test needs.)
    #[doc(hidden)]
    pub fn set_state_for_test(&mut self, state: LocalModelsState) {
        self.state = Loadable::loaded(state);
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

    /// Models currently loaded (ready to serve) in the managed local store.
    pub fn loaded_models(&self) -> Vec<LocalModelInfo> {
        self.models()
            .iter()
            .filter(|m| matches!(m.status, LocalModelStatus::Loaded { .. }))
            .cloned()
            .collect()
    }

    /// The managed store's *selectable* models — everything on disk
    /// (loaded, warming, or not-yet-loaded; a request loads on demand).
    /// The "On this device" group of the model picker.
    ///
    /// `on_disk` is the whole test: a row is not always a file. One still
    /// downloading has only a `.part`, and a failed download leaves a row
    /// carrying nothing but its error — picking either could only fail at
    /// load time. Both stay in [`Self::models`], where Settings shows the
    /// progress or the error.
    pub fn selectable_models(&self) -> Vec<LocalModelInfo> {
        self.models()
            .iter()
            .filter(|m| m.on_disk)
            .cloned()
            .collect()
    }

    /// The configured llamacpp backends' scanned directories (empty unless
    /// loaded).
    pub fn external(&self) -> &[eidola_app_core::ExternalEngineBackend] {
        self.state
            .value()
            .map(|s| s.external.as_slice())
            .unwrap_or(&[])
    }

    /// One llamacpp backend's scanned models (empty when absent).
    pub fn external_models(&self, backend_id: &str) -> Vec<LocalModelInfo> {
        self.external()
            .iter()
            .find(|b| b.backend_id == backend_id)
            .map(|b| b.models.clone())
            .unwrap_or_default()
    }

    /// One llamacpp backend's currently loaded models — its group in the
    /// model picker.
    pub fn external_loaded_models(&self, backend_id: &str) -> Vec<LocalModelInfo> {
        self.external_models(backend_id)
            .into_iter()
            .filter(|m| matches!(m.status, LocalModelStatus::Loaded { .. }))
            .collect()
    }

    /// One llamacpp backend's *selectable* models — everything the scan
    /// actually found (a request loads on demand); its group in the model
    /// picker. Same `on_disk` test as [`Self::selectable_models`].
    pub fn external_selectable_models(&self, backend_id: &str) -> Vec<LocalModelInfo> {
        self.external_models(backend_id)
            .into_iter()
            .filter(|m| m.on_disk)
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

    /// Whether this URL's download is **being started** right now — the keyed
    /// op slot, which is what "in flight" means here (STATE.md: no shared busy
    /// flags; in-flight is the presence of the task).
    ///
    /// It covers the initiating call only. Once app-core has accepted the
    /// transfer the snapshot's own row carries it (`Downloading`, with Cancel),
    /// which is the state every surface reads from there on.
    pub fn download_pending(&self, url: &str) -> bool {
        self.op_tasks.contains_key(&Self::download_key(url))
    }

    /// The pending slot's key for a download URL.
    ///
    /// **It is the identity app-core deduplicates on, never the spelling a
    /// door happened to hold** — `resolve_model_download` maps every
    /// equivalent spelling of one model (a Hugging Face `/blob/` page and its
    /// `/resolve/` object, a `?download=true` suffix, any two URLs naming the
    /// same file) onto the one slug app-core keys its transfer map by. The two
    /// **must** agree: a key that separates what app-core joins lets a second
    /// door miss the pending lookup, reach app-core, and publish its
    /// "already downloading" refusal over the transfer that is working —
    /// exactly the race [`Self::download`]'s non-reentrancy exists to prevent.
    /// Every read and write of the slot goes through here, so there is one
    /// derivation and no place for a raw URL to slip back in.
    ///
    /// A URL app-core would reject (not a `.gguf`, not http) has no transfer
    /// identity to key on, so the trimmed text stands in — enough to make a
    /// repeat of that very paste a no-op while the first is being refused, and
    /// the refusal is what the reader wants either way. The `?` prefix cannot
    /// collide with a slug, whose characters app-core restricts.
    fn download_key(url: &str) -> String {
        match eidola_app_core::resolve_model_download(url) {
            Ok(target) => format!("download:{}", target.slug),
            Err(_) => format!("download:?{}", url.trim()),
        }
    }

    /// Start a model download (curated entry or pasted URL). The transfer
    /// itself is core-owned; progress arrives via `Change::LocalModels`.
    ///
    /// **Not re-entrant for a transfer that is already being started**
    /// ([`Self::download_key`] decides what "the same transfer" means, and
    /// answers as app-core does). Several surfaces open onto this one
    /// operation — a failed row's Retry, the curated catalog's Download, the
    /// paste-a-URL row — and they all ask for the *same* thing, so a second
    /// call is one request twice rather than a newer intent. [`Self::run_op`]'s keep-newest
    /// supersede is the wrong shape for that: the second `op_tasks.insert`
    /// drops the first continuation while the core call it issued runs on, and
    /// app-core then refuses the duplicate ("already downloading") — a failure
    /// published over a transfer that is working, and a slot that clears while
    /// the real transfer is still being started.
    ///
    /// Re-entry is therefore a **no-op**, which is the shape this codebase
    /// already takes wherever an operation has exactly one meaning
    /// (`UpdateStore::check_now`'s `checking` flag; `Space`'s exclusive
    /// `post_runner`). The chained last-wins of the keyed *edit* slots
    /// (`AgentsStore`, `ParticipantsStore`) is for the opposite case — there a
    /// successor carries a newer value that must win, and here there is no
    /// newer value to carry.
    ///
    /// It covers the initiating call only, which is all the slot knows about:
    /// once it clears, the snapshot's own `Downloading` row (with Cancel) is
    /// what stands in the way, and a press landing after that reaches
    /// app-core's own honest refusal rather than superseding anything.
    pub fn download(&mut self, url: String, cx: &mut Context<Self>) {
        if self.download_pending(&url) {
            return;
        }
        let key = Self::download_key(&url);
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

    /// Forget a standing download/load failure. Clears the report, nothing
    /// else — where the failure is all that is left of a download, the row it
    /// synthesizes goes with it.
    pub fn dismiss_failure(&mut self, id: String, cx: &mut Context<Self>) {
        let key = format!("dismiss:{id}");
        self.run_op(key, cx, move |c| async move {
            c.dismiss_local_model_failure(id).await
        });
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

    /// Pin/unpin a loaded model's engine (protection from automatic LRU
    /// unloading; manual unload still applies).
    pub fn set_pinned(&mut self, id: String, pinned: bool, cx: &mut Context<Self>) {
        let key = format!("pin:{id}");
        self.run_op(key, cx, move |c| async move {
            c.set_local_model_pinned(id, pinned).await
        });
    }

    /// Shared operation shape: clear the standing error, run the initiating
    /// core call in this operation's keyed slot, and surface a failure in
    /// `op_error`. Success needs no local action — the bus-driven refresh
    /// carries the new state.
    ///
    /// **Notify after the slot is occupied**, not before. A paint that runs
    /// off the notify must see `download_pending` (and its siblings) as true,
    /// or a failed row's Retry is still in the tree for a second press.
    fn run_op<F, Fut>(&mut self, key: String, cx: &mut Context<Self>, f: F)
    where
        F: FnOnce(Arc<AppCore>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), eidola_app_core::error::AppError>> + Send + 'static,
    {
        self.op_error = None;
        let Some(core) = self.app_core.clone() else {
            cx.notify();
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
        cx.notify();
    }
}
