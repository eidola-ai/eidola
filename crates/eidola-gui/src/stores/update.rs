//! `UpdateStore` — the verified update-notification state. Migrated almost
//! verbatim from the old `Core`'s update fields; this flow was already
//! doctrine-shaped (its own per-operation in-flight flag, never the shared
//! busy gate), so it is the pattern everything else converges to.

use std::sync::Arc;

use eidola_app_core::AppCore;
use eidola_app_core::updates::UpdateCheckSnapshot;
use gpui::{Context, Task};

pub struct UpdateStore {
    app_core: Option<Arc<AppCore>>,
    /// Outcome of the most recent completed check (manual or background poll).
    snapshot: Option<UpdateCheckSnapshot>,
    /// True while a manual check is in flight. This is a *per-operation*
    /// in-flight flag, not a shared busy gate — it gates only this store's
    /// own re-entry.
    checking: bool,
    /// Supersede slot for a manual check.
    task: Option<Task<()>>,
}

impl UpdateStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            snapshot: None,
            checking: false,
            task: None,
        }
    }

    /// A stub store with a fixture snapshot / in-flight flag (tests).
    pub fn stub(snapshot: Option<UpdateCheckSnapshot>, checking: bool) -> Self {
        Self {
            app_core: None,
            snapshot,
            checking,
            task: None,
        }
    }

    pub fn snapshot(&self) -> Option<&UpdateCheckSnapshot> {
        self.snapshot.as_ref()
    }

    pub fn checking(&self) -> bool {
        self.checking
    }

    /// Re-read the persisted last-check snapshot (bus-driven on
    /// `Change::UpdateState`, and on construction so a poll result that landed
    /// while no window was open is reflected on next open).
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        if let Some(core) = self.app_core.as_ref() {
            self.snapshot = core.last_update_check();
            cx.notify();
        }
    }

    /// Pull the persisted last-check snapshot into the cache (alias of
    /// `refresh`, kept for the view's construction-time call).
    pub fn load_last(&mut self, cx: &mut Context<Self>) {
        self.refresh(cx);
    }

    /// Run a manual update check. Its own `checking` flag makes re-entry a
    /// no-op while one is in flight (so the constructor-triggered check and a
    /// menu re-check are naturally idempotent).
    pub fn check_now(&mut self, cx: &mut Context<Self>) {
        if self.checking {
            return;
        }
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.checking = true;
        cx.notify();
        self.task = Some(cx.spawn(async move |this, cx| {
            let snapshot = core.update_check().await;
            let _ = this.update(cx, |this, cx| {
                this.checking = false;
                this.snapshot = Some(snapshot);
                this.task = None;
                cx.notify();
            });
        }));
    }

    /// Record the user's explicit "treat as update" decision for a
    /// claims-changed release, then refresh the cached snapshot.
    pub fn accept_changed_claims(
        &mut self,
        version: String,
        manifest_sha256: String,
        cx: &mut Context<Self>,
    ) {
        let Some(core) = self.app_core.as_ref() else {
            return;
        };
        if core.accept_changed_claims(version, manifest_sha256).is_ok() {
            self.snapshot = core.last_update_check();
        }
        cx.notify();
    }

    /// Start the app-core background poll loop (launch + every ~6h).
    /// Idempotent; no-op on a stub.
    pub fn start_polling(&self) {
        if let Some(core) = self.app_core.as_ref() {
            core.start_update_polling();
        }
    }
}
