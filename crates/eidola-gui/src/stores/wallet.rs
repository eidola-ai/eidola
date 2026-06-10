//! `WalletStore` — the credential lifecycle listing and in-flight recovery.
//!
//! Owns two cells:
//! - `lifecycle` — every credential with its computed state (active /
//!   spending / spent / expired), the Wallet pane's listing.
//! - `credentials` — the active (spendable) credentials, which the chat
//!   onboarding stage consults to decide whether to show the plans page.
//!
//! Both are refreshed together (one `wallet_lifecycle` + one
//! `wallet_credentials` call) on `Change::Wallet` and at startup. Startup
//! recovery (`recover`) moves here from `lib.rs::run()`.

use std::sync::Arc;

use eidola_app_core::{AppCore, CredentialInfo, CredentialLifecycleInfo};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct WalletStore {
    app_core: Option<Arc<AppCore>>,
    lifecycle: Loadable<Vec<CredentialLifecycleInfo>>,
    credentials: Loadable<Vec<CredentialInfo>>,
    /// Supersede slot for the combined refresh.
    task: Option<Task<()>>,
    /// Slot for an in-flight recovery (its own slot so it doesn't cancel a
    /// listing refresh).
    recover_task: Option<Task<()>>,
}

impl WalletStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            lifecycle: Loadable::NotLoaded,
            credentials: Loadable::NotLoaded,
            task: None,
            recover_task: None,
        }
    }

    /// A stub store with fixture lifecycle + active-credential lists (tests).
    pub fn stub(lifecycle: Vec<CredentialLifecycleInfo>, credentials: Vec<CredentialInfo>) -> Self {
        Self {
            app_core: None,
            lifecycle: if lifecycle.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(lifecycle)
            },
            credentials: if credentials.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(credentials)
            },
            task: None,
            recover_task: None,
        }
    }

    // -- Reads --------------------------------------------------------------

    pub fn lifecycle(&self) -> &Loadable<Vec<CredentialLifecycleInfo>> {
        &self.lifecycle
    }

    /// The lifecycle rows as a slice.
    pub fn lifecycle_rows(&self) -> &[CredentialLifecycleInfo] {
        self.lifecycle.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// The active (spendable) credentials as a slice.
    pub fn credentials(&self) -> &[CredentialInfo] {
        self.credentials
            .value()
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// True while either listing cell is doing an initial load or is stale —
    /// the wallet pane's "Loading…" hint.
    pub fn is_loading(&self) -> bool {
        self.lifecycle.is_loading() || self.lifecycle.is_stale() || self.recover_task.is_some()
    }

    // -- Refresh ------------------------------------------------------------

    /// Refresh both the lifecycle listing and the active-credential list in
    /// one task. Fire-and-notify supersede slot.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.lifecycle = std::mem::take(&mut self.lifecycle).to_loading();
        self.credentials = std::mem::take(&mut self.credentials).to_loading();
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move {
                let lifecycle = c.wallet_lifecycle().await?;
                let active = c.wallet_credentials().await?;
                Ok((lifecycle, active))
            })
            .await;
            let _ = this.update(cx, |this, cx| {
                match result {
                    Ok((lifecycle, active)) => {
                        this.lifecycle = Loadable::loaded(lifecycle);
                        this.credentials = Loadable::loaded(active);
                    }
                    Err(e) => {
                        this.lifecycle =
                            std::mem::take(&mut this.lifecycle).resolve(Err(e.clone()));
                        this.credentials = std::mem::take(&mut this.credentials).resolve(Err(e));
                    }
                }
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Attempt recovery of any in-flight credentials, then refresh the
    /// listings. `on_done` receives the recovered nonces so the caller can
    /// surface a result message. The whole flow is owned by `recover_task`.
    pub fn recover(
        &mut self,
        cx: &mut Context<Self>,
        on_done: impl FnOnce(&mut WalletStore, Vec<String>, &mut Context<WalletStore>) + 'static,
    ) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.recover_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move {
                let recovered = c.recover_spending_credentials().await?;
                let lifecycle = c.wallet_lifecycle().await?;
                let active = c.wallet_credentials().await?;
                Ok((recovered, lifecycle, active))
            })
            .await;
            let _ = this.update(cx, |this, cx| {
                this.recover_task = None;
                if let Ok((recovered, lifecycle, active)) = result {
                    this.lifecycle = Loadable::loaded(lifecycle);
                    this.credentials = Loadable::loaded(active);
                    cx.notify();
                    on_done(this, recovered, cx);
                } else {
                    cx.notify();
                }
            });
        }));
        cx.notify();
    }
}
