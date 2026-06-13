//! `AccountStore` — balances, prices, and account lifecycle (create / reset /
//! checkout). Per `docs/architecture/state.md` the checkout *poll* is
//! view-owned (it dies with its window); its result lands here via the bus.
//!
//! Two method shapes, distinguished by who owns the task:
//!
//! - `refresh_balances` / `refresh_prices` — **fire-and-notify** supersede
//!   slots. The store owns the task; callers observe the `Loadable`.
//! - `request_checkout` / `request_account_create` / `request_balances` —
//!   **awaitable**. They return a `oneshot::Receiver`; the *caller* awaits it
//!   inside the caller's own slot, so the work dies with the caller (a
//!   checkout poll dies with its window). The durable write still happens
//!   core-side and the bus still emits.

use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{AccountCreateResult, AppCore, BalancesResult, PriceInfo};
use gpui::{Context, Task};
use tokio::sync::oneshot;

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct AccountStore {
    app_core: Option<Arc<AppCore>>,
    balances: Loadable<BalancesResult>,
    prices: Loadable<Vec<PriceInfo>>,
    /// Supersede slots — replacing cancels the predecessor. `Loading` on a
    /// cell implies its slot is `Some`.
    balances_task: Option<Task<()>>,
    prices_task: Option<Task<()>>,
    /// Slot for the fire-and-notify `create_account` op (its own slot so it
    /// never cancels a balances/prices refresh).
    lifecycle_task: Option<Task<()>>,
    /// The last account-lifecycle write error (create / reset), or `None`.
    /// Honest-states rule: a failed Settings button must say so. Cleared at the
    /// start of the next attempt and on success; rendered by `AccountView`.
    account_op_error: Option<AppError>,
}

impl AccountStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            balances: Loadable::NotLoaded,
            prices: Loadable::NotLoaded,
            balances_task: None,
            prices_task: None,
            lifecycle_task: None,
            account_op_error: None,
        }
    }

    /// A stub store with fixture balances/prices (tests).
    pub fn stub(balances: Option<BalancesResult>, prices: Vec<PriceInfo>) -> Self {
        Self {
            app_core: None,
            balances: match balances {
                Some(b) => Loadable::loaded(b),
                None => Loadable::NotLoaded,
            },
            prices: if prices.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(prices)
            },
            balances_task: None,
            prices_task: None,
            lifecycle_task: None,
            account_op_error: None,
        }
    }

    // -- Reads --------------------------------------------------------------

    pub fn balances(&self) -> &Loadable<BalancesResult> {
        &self.balances
    }

    pub fn prices(&self) -> &Loadable<Vec<PriceInfo>> {
        &self.prices
    }

    /// True while either cell is doing an initial load — the panes' "Loading…"
    /// hint (replaces the old shared `busy` flag for account surfaces).
    pub fn is_loading(&self) -> bool {
        self.balances.is_loading()
            || self.prices.is_loading()
            || self.balances.is_stale()
            || self.prices.is_stale()
    }

    // -- Refresh (fire-and-notify) -----------------------------------------

    pub fn refresh_balances(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.balances = std::mem::take(&mut self.balances).to_loading();
        self.balances_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.account_balances().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.balances = std::mem::take(&mut this.balances).resolve(result);
                this.balances_task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    pub fn refresh_prices(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.prices = std::mem::take(&mut self.prices).to_loading();
        self.prices_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.account_prices().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.prices = std::mem::take(&mut this.prices).resolve(result);
                this.prices_task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Directly set the balances snapshot (used by the view-owned checkout
    /// poll, which fetches balances inside its own window task and writes the
    /// result back here — outside the bus, since the poll is the initiator).
    pub fn set_balances(&mut self, balances: BalancesResult, cx: &mut Context<Self>) {
        self.balances = Loadable::loaded(balances);
        self.balances_task = None;
        cx.notify();
    }

    // -- Awaitable (caller owns the task) ----------------------------------

    /// Create a checkout session for `price_id`; the caller awaits the
    /// returned receiver inside its own task. `None` on a stub.
    pub fn request_checkout(
        &self,
        price_id: String,
    ) -> Option<oneshot::Receiver<Result<String, AppError>>> {
        let core = self.app_core.clone()?;
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let _ = tx.send(core.account_checkout(price_id).await);
        });
        Some(rx)
    }

    /// Create an anonymous account; the caller (onboarding flow) awaits the
    /// returned receiver inside its own task and refreshes config on success.
    /// `None` on a stub.
    pub fn request_account_create(
        &self,
    ) -> Option<oneshot::Receiver<Result<AccountCreateResult, AppError>>> {
        let core = self.app_core.clone()?;
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let _ = tx.send(core.account_create().await);
        });
        Some(rx)
    }

    /// Fetch balances; the caller (checkout poll) awaits inside its own loop.
    /// `None` on a stub.
    pub fn request_balances(&self) -> Option<oneshot::Receiver<Result<BalancesResult, AppError>>> {
        let core = self.app_core.clone()?;
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let _ = tx.send(core.account_balances().await);
        });
        Some(rx)
    }

    // -- Account lifecycle writes ------------------------------------------

    /// The last account-lifecycle (create / reset) error, if the most recent
    /// attempt failed. Cleared at the start of the next attempt and on success.
    pub fn account_op_error(&self) -> Option<&AppError> {
        self.account_op_error.as_ref()
    }

    /// Test-only: set the account-op error directly so a behavior test can
    /// render the failure banner without a failing backend.
    #[doc(hidden)]
    pub fn set_account_op_error_for_test(
        &mut self,
        error: Option<AppError>,
        cx: &mut Context<Self>,
    ) {
        self.account_op_error = error;
        cx.notify();
    }

    /// Create an account as a fire-and-notify store op (the Account settings
    /// pane's "Create account" button). Refreshes prices+balances on success;
    /// on a real backend it also emits `Change::Config`/`Change::Account`.
    /// On failure the error is **stored** (honest-states rule — the button must
    /// not silently do nothing) and rendered by `AccountView`.
    pub fn create_account(&mut self, cx: &mut Context<Self>) {
        // Clear any prior error at the start of this attempt.
        self.account_op_error = None;
        cx.notify();
        let Some(core) = self.app_core.clone() else {
            return;
        };
        let task = cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.account_create().await }).await;
            let _ = this.update(cx, |this, cx| {
                match result {
                    Ok(_) => {
                        this.account_op_error = None;
                        this.refresh_prices(cx);
                        this.refresh_balances(cx);
                    }
                    Err(e) => {
                        // Surface the failure instead of dropping it.
                        this.account_op_error = Some(e);
                        cx.notify();
                    }
                }
            });
        });
        // Own the task in its own slot (not `.detach()`): it dies with the
        // store, never cancels a balances/prices refresh, and the result drives
        // those refreshes from inside its body.
        self.lifecycle_task = Some(task);
        cx.notify();
    }

    /// Reset the account (forget local keys). Synchronous core write; refreshes
    /// the local cells to their now-empty state on the next bus tick. A failed
    /// reset is stored (same honest-states treatment as create) and rendered.
    pub fn reset_account(&mut self, cx: &mut Context<Self>) {
        self.account_op_error = None;
        let Some(core) = self.app_core.as_ref() else {
            cx.notify();
            return;
        };
        match core.reset_account() {
            Ok(()) => {
                self.balances = Loadable::NotLoaded;
                self.balances_task = None;
            }
            Err(e) => {
                self.account_op_error = Some(e);
            }
        }
        cx.notify();
    }
}
