//! `ConfigStore` — the synchronous, write-through projection of app-core's
//! config. Owns the `ConfigState` snapshot and all config mutations
//! (base URL, default model, attestation URL, …). Reads are synchronous;
//! writes go straight through to `AppCore` and re-read the snapshot. On a
//! real backend each write also emits `Change::Config`, so other windows'
//! `ConfigStore`s refresh via the bus.
//!
//! Per `docs/architecture/state.md`: `ConfigState` is not a `Loadable` — it is
//! always present (read synchronously at construction) on a backed store, and
//! `None` only on a stub. The store keeps a `last_error` for write failures so
//! the settings panes can surface them.

use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ConfigState};
use gpui::Context;

pub struct ConfigStore {
    app_core: Option<Arc<AppCore>>,
    /// The current config snapshot. `Some` on a backed store (seeded at
    /// construction, refreshed on `Change::Config`); `None` on a stub until a
    /// test installs one.
    state: Option<ConfigState>,
    /// The last config-write error, surfaced by the settings panes.
    error: Option<AppError>,
}

impl ConfigStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        let state = app_core.as_ref().map(|c| c.config_state());
        Self {
            app_core,
            state,
            error: None,
        }
    }

    /// A stub store for tests, with no backend and the given fixture state.
    pub fn stub(state: Option<ConfigState>) -> Self {
        Self {
            app_core: None,
            state,
            error: None,
        }
    }

    /// The current config snapshot, if known.
    pub fn state(&self) -> Option<&ConfigState> {
        self.state.as_ref()
    }

    /// Test-only: install a fixture config snapshot.
    #[doc(hidden)]
    pub fn set_state_for_test(&mut self, state: Option<ConfigState>) {
        self.state = state;
    }

    /// The last config-write error, if any.
    pub fn error(&self) -> Option<&AppError> {
        self.error.as_ref()
    }

    pub fn clear_error(&mut self, cx: &mut Context<Self>) {
        if self.error.take().is_some() {
            cx.notify();
        }
    }

    /// Re-read the snapshot from the backend (bus-driven on `Change::Config`,
    /// and called by views after a config-mutating op completes elsewhere).
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        if let Some(core) = self.app_core.as_ref() {
            self.state = Some(core.config_state());
            cx.notify();
        }
    }

    fn write(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&AppCore) -> Result<(), AppError>) {
        let Some(core) = self.app_core.as_ref() else {
            return;
        };
        match f(core) {
            Ok(()) => {
                self.state = Some(core.config_state());
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
        cx.notify();
    }

    pub fn set_base_url(&mut self, url: String, cx: &mut Context<Self>) {
        self.write(cx, |c| c.set_base_url(url));
    }

    pub fn clear_base_url_override(&mut self, cx: &mut Context<Self>) {
        self.write(cx, |c| c.clear_base_url_override());
    }

    pub fn set_default_model(&mut self, model: String, cx: &mut Context<Self>) {
        self.write(cx, |c| c.set_default_model(model));
    }

    #[allow(dead_code)]
    pub fn set_attestation_url(&mut self, url: String, cx: &mut Context<Self>) {
        self.write(cx, |c| c.set_attestation_url(url));
    }

    #[allow(dead_code)]
    pub fn set_account_credentials(&mut self, id: String, secret: String, cx: &mut Context<Self>) {
        self.write(cx, |c| c.set_account_credentials(id, secret));
    }
}
