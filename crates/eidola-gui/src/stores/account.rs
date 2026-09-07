//! `AccountStore` — balances, prices, the required-terms snapshot, and
//! account lifecycle (create / reset / checkout). Per
//! `crates/eidola-gui/STATE.md` the checkout *poll* is
//! view-owned (it dies with its window); its result lands here via the bus.
//!
//! Two method shapes, distinguished by who owns the task:
//!
//! - `refresh_balances` / `refresh_prices` / `refresh_terms` —
//!   **fire-and-notify** supersede
//!   slots. The store owns the task; callers observe the `Loadable`.
//! - `request_checkout` / `request_account_create` / `request_balances` —
//!   **awaitable**. They return a `oneshot::Receiver`; the *caller* awaits it
//!   inside the caller's own slot, so the work dies with the caller (a
//!   checkout poll dies with its window). The durable write still happens
//!   core-side and the bus still emits.

use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AccountCreateResult, AppCore, BalancesResult, PriceInfo, SubscriptionInfo, TermsDocument,
};
use gpui::{Context, Task};
use tokio::sync::oneshot;

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct AccountStore {
    app_core: Option<Arc<AppCore>>,
    balances: Loadable<BalancesResult>,
    prices: Loadable<Vec<PriceInfo>>,
    /// The account's subscription standing. Core-side this is a live read
    /// that persists nothing, so nothing on the bus ever invalidates it —
    /// it is refreshed when a surface that shows it opens, and by that
    /// surface's own retry.
    subscription: Loadable<SubscriptionInfo>,
    /// The document versions the server currently requires acceptance of —
    /// **the snapshot a consent surface renders and then hands back to
    /// `request_account_create`**, so what is agreed to is what was shown.
    /// Empty `Loaded` means the server runs no acceptance gate.
    terms: Loadable<Vec<TermsDocument>>,
    /// Supersede slots — replacing cancels the predecessor. `Loading` on a
    /// cell implies its slot is `Some`.
    balances_task: Option<Task<()>>,
    prices_task: Option<Task<()>>,
    subscription_task: Option<Task<()>>,
    terms_task: Option<Task<()>>,
    /// The last account-lifecycle write error (today: reset), or `None`.
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
            subscription: Loadable::NotLoaded,
            terms: Loadable::NotLoaded,
            balances_task: None,
            prices_task: None,
            subscription_task: None,
            terms_task: None,
            account_op_error: None,
        }
    }

    /// A stub store with fixture balances/prices/subscription (tests).
    pub fn stub(
        balances: Option<BalancesResult>,
        prices: Vec<PriceInfo>,
        subscription: Option<SubscriptionInfo>,
    ) -> Self {
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
            subscription: match subscription {
                Some(s) => Loadable::loaded(s),
                None => Loadable::NotLoaded,
            },
            terms: Loadable::NotLoaded,
            balances_task: None,
            prices_task: None,
            subscription_task: None,
            terms_task: None,
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

    pub fn subscription(&self) -> &Loadable<SubscriptionInfo> {
        &self.subscription
    }

    /// The document versions a consent surface must show before an account is
    /// created. See [`AccountStore::request_account_create`].
    pub fn terms(&self) -> &Loadable<Vec<TermsDocument>> {
        &self.terms
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

    /// Re-read the account's subscription standing. Only meaningful with an
    /// account configured; callers gate on that the way they gate balances.
    pub fn refresh_subscription(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.subscription = std::mem::take(&mut self.subscription).to_loading();
        self.subscription_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.account_subscription().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.subscription = std::mem::take(&mut this.subscription).resolve(result);
                this.subscription_task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Re-read the document versions the server requires acceptance of.
    ///
    /// A consent surface calls this when it opens and renders the resulting
    /// snapshot; the *same* snapshot is what
    /// [`AccountStore::request_account_create`] then submits. Public data —
    /// no account needed, and nothing here is account-scoped.
    pub fn refresh_terms(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.terms = std::mem::take(&mut self.terms).to_loading();
        self.terms_task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.current_terms().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.terms = std::mem::take(&mut this.terms).resolve(result);
                this.terms_task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Test-only: put the terms cell in an arbitrary state so a behavior test
    /// can render a consent surface without a backend.
    #[doc(hidden)]
    pub fn set_terms_for_test(
        &mut self,
        cell: Loadable<Vec<TermsDocument>>,
        cx: &mut Context<Self>,
    ) {
        self.terms = cell;
        cx.notify();
    }

    /// Drop everything this store holds that describes *a particular
    /// account*, so an identity switch can never leave one account's answer
    /// on screen under another's name.
    ///
    /// The subscription is the whole set today: prices are a public catalog,
    /// and balances are written from the verified account's own response
    /// (or cleared beside this on reset). Clearing rather than refreshing is
    /// deliberate — showing nothing while a surface re-reads is honest,
    /// where showing the previous account's standing is not, and on a
    /// billing surface that is the difference that matters.
    pub fn forget_account_scoped_state(&mut self, cx: &mut Context<Self>) {
        self.subscription = Loadable::NotLoaded;
        self.subscription_task = None;
        cx.notify();
    }

    /// The configured account is now a *different* account: drop what the
    /// old one owned and read the new one's standing.
    ///
    /// **Called on the commit, never on the attempt.** Creating and linking
    /// both refuse outright when credentials already exist, so clearing when
    /// the request went out blanked the cell for an operation that never
    /// happened — an Account pane already open then lost its billing section
    /// entirely, with no error branch to put it back. Clearing *and*
    /// refreshing together is the point: a bare refresh would leave the
    /// previous account's value on screen as `Loaded { stale }` while the
    /// new read is in flight, which on a billing surface is the wrong
    /// account's answer wearing the right account's name.
    pub fn account_identity_changed(&mut self, cx: &mut Context<Self>) {
        self.forget_account_scoped_state(cx);
        self.refresh_subscription(cx);
    }

    /// The account this machine speaks for **right now**, read straight from
    /// app-core's config.
    ///
    /// Deliberately not `ConfigStore`: that snapshot is a *cache* refreshed
    /// from the bus, and every write here commits to config synchronously and
    /// then emits `Change::Config`, so between the commit and the next bus
    /// tick the cached id still names the account that has just been replaced.
    /// Anything deciding whether a credential-bearing round trip still belongs
    /// to the configured account has to read past that lag, or it is asking
    /// the stale copy about the staleness it exists to detect.
    ///
    /// `fallback` answers on a stub store, which has no config to read.
    pub fn account_identity(&self, fallback: Option<&str>) -> Option<String> {
        match self.app_core.as_ref() {
            Some(core) => core.config_state().account_id,
            None => fallback.map(str::to_string),
        }
    }

    /// Whether a URL minted while `minted_for` was configured may still be
    /// opened — that is, whether the configured account is still that one.
    pub fn mint_is_current(&self, minted_for: Option<&str>, fallback: Option<&str>) -> bool {
        self.account_identity(fallback).as_deref() == minted_for
    }

    /// Test-only: put the subscription cell in an arbitrary state, so a
    /// behavior test or driver scene can render "checking" and "couldn't
    /// check" without a backend that stalls or fails on cue.
    #[doc(hidden)]
    pub fn set_subscription_for_test(
        &mut self,
        cell: Loadable<SubscriptionInfo>,
        cx: &mut Context<Self>,
    ) {
        self.subscription = cell;
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
            // The view keeps its own before-and-after identity guard, so
            // only the link is passed on.
            let _ = tx.send(core.account_checkout(price_id).await.map(|m| m.url));
        });
        Some(rx)
    }

    /// Create an anonymous account, recording its acceptance of
    /// `accepted_terms` — **the snapshot the caller rendered**, read from
    /// [`AccountStore::terms`]. The caller (the consent flow) awaits the
    /// returned receiver inside its own task and refreshes config on success.
    /// `None` on a stub.
    ///
    /// The snapshot is a parameter rather than something app-core re-reads,
    /// so a document version advancing while the consent screen was open is
    /// refused by the server instead of silently recorded — see
    /// `AppCore::account_create`.
    pub fn request_account_create(
        &self,
        accepted_terms: Vec<TermsDocument>,
    ) -> Option<oneshot::Receiver<Result<AccountCreateResult, AppError>>> {
        let core = self.app_core.clone()?;
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let _ = tx.send(core.account_create(accepted_terms).await);
        });
        Some(rx)
    }

    /// Verify an *existing* account entered by the user: set its credentials,
    /// then fetch its balance. On success the credentials stay configured (the
    /// user has now linked their account); on failure they are rolled back so a
    /// bad ID/secret attempt doesn't strand broken credentials in the config.
    /// The caller (onboarding "existing account" slide) awaits the returned
    /// receiver in its own task slot. `None` on a stub.
    pub fn request_verify_account(
        &self,
        id: String,
        secret: String,
    ) -> Option<oneshot::Receiver<Result<BalancesResult, AppError>>> {
        let core = self.app_core.clone()?;
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let result = async {
                core.set_account_credentials(id, secret)?;
                match core.account_balances().await {
                    Ok(balances) => Ok(balances),
                    Err(e) => {
                        // Undo the just-written credentials so the config is
                        // unchanged after a failed verification attempt.
                        let _ = core.reset_account();
                        Err(e)
                    }
                }
            }
            .await;
            let _ = tx.send(result);
        });
        Some(rx)
    }

    /// Mint a billing-portal session for the click that is about to open it.
    /// The link is short-lived, so it is asked for at the moment of use
    /// rather than held from whenever the pane last read — the same shape as
    /// `request_checkout`. The caller awaits inside its own slot. `None` on a
    /// stub.
    pub fn request_portal(&self) -> Option<oneshot::Receiver<Result<String, AppError>>> {
        let core = self.app_core.clone()?;
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let _ = tx.send(core.account_portal().await);
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

    /// The last account-lifecycle error, if the most recent attempt failed.
    /// Cleared at the start of the next attempt and on success.
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

    /// Reset the account (forget local keys). Synchronous core write; refreshes
    /// the local cells to their now-empty state on the next bus tick. A failed
    /// reset is stored (the honest-states treatment) and rendered.
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
                // The subscription belonged to the account whose keys were
                // just forgotten; keeping it on screen would attribute
                // someone's subscription to nobody.
                self.forget_account_scoped_state(cx);
            }
            Err(e) => {
                self.account_op_error = Some(e);
            }
        }
        cx.notify();
    }
}
