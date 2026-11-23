//! The account and the wallet, as the phone shows them: the same reads and
//! writes as the desktop app's account and wallet stores
//! (`crates/eidola-gui/src/stores/account.rs`, `wallet.rs`), cut to what the
//! phone offers.
//!
//! - **The configured account** is a snapshot of the profile's config
//!   ([`AccountStore::config`]), re-read on `Change::Config`. It is a cache:
//!   anything deciding whether a credential-bearing round trip still belongs
//!   to the account configured *now* asks [`AccountStore::mint_is_current`],
//!   which reads app-core's config past it (the desktop's rule).
//! - **Balances, prices, the subscription and the terms** are live server
//!   reads, each a [`Loadable`] with its own supersede slot. Nothing on the
//!   bus invalidates them; the screens that show them ask on every visit.
//!   A purchase finishes in the browser, so the first time the app comes back
//!   to the foreground after a checkout link went out, the balance is read
//!   again ([`AccountStore::returned_to_foreground`]).
//! - **State that describes an account does not outlive it**: creating,
//!   linking and resetting all drop the subscription and the balance
//!   ([`AccountStore::identity_changed`], [`AccountStore::reset`]).
//! - **The wallet's lifecycle** ([`WalletStore`]) is read-only here, re-read
//!   on `Change::Wallet` and `Change::Account`.

use std::rc::Rc;

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AccountCreateResult, AppCore, BalancesResult, CheckoutMint, ConfigState,
    CredentialLifecycleInfo, PortalMint, PriceInfo, SubscriptionInfo, TermsDocument,
};
use gpui::{Context, Task};

use super::loadable::Loadable;

/// The backend id the account belongs to.
pub const EIDOLA_BACKEND: &str = "eidola";

pub struct AccountStore {
    core: Rc<AppCore>,
    config: ConfigState,
    balances: Loadable<BalancesResult>,
    balances_task: Option<Task<()>>,
    prices: Loadable<Vec<PriceInfo>>,
    prices_task: Option<Task<()>>,
    subscription: Loadable<SubscriptionInfo>,
    subscription_task: Option<Task<()>>,
    terms: Loadable<Vec<TermsDocument>>,
    terms_task: Option<Task<()>>,
    /// Whether the Eidola backend is enabled (`None` until read). Choosing to
    /// go on without an account disables it, which is what keeps the first
    /// run from opening again.
    eidola_enabled: Loadable<bool>,
    backends_task: Option<Task<()>>,
    /// The last account write that was refused (a reset).
    op_error: Option<AppError>,
    /// A checkout link went to the browser and the balance has not been
    /// read since the app came back.
    checkout_out: bool,
}

impl AccountStore {
    pub fn new(core: Rc<AppCore>, cx: &mut Context<Self>) -> Self {
        let config = core.config_state();
        let mut store = Self {
            core,
            config,
            balances: Loadable::NotLoaded,
            balances_task: None,
            prices: Loadable::NotLoaded,
            prices_task: None,
            subscription: Loadable::NotLoaded,
            subscription_task: None,
            terms: Loadable::NotLoaded,
            terms_task: None,
            eidola_enabled: Loadable::NotLoaded,
            backends_task: None,
            op_error: None,
            checkout_out: false,
        };
        store.refresh_backends(cx);
        store
    }

    pub fn config(&self) -> &ConfigState {
        &self.config
    }

    pub fn has_account(&self) -> bool {
        self.config.has_account && self.config.has_account_secret
    }

    pub fn balances(&self) -> &Loadable<BalancesResult> {
        &self.balances
    }

    pub fn prices(&self) -> &Loadable<Vec<PriceInfo>> {
        &self.prices
    }

    pub fn subscription(&self) -> &Loadable<SubscriptionInfo> {
        &self.subscription
    }

    pub fn terms(&self) -> &Loadable<Vec<TermsDocument>> {
        &self.terms
    }

    pub fn eidola_enabled(&self) -> &Loadable<bool> {
        &self.eidola_enabled
    }

    pub fn op_error(&self) -> Option<&AppError> {
        self.op_error.as_ref()
    }

    /// The config changed: re-read the snapshot. Credentials that changed
    /// under the store (from anywhere) take the account's state with them.
    pub fn on_config_changed(&mut self, cx: &mut Context<Self>) {
        let config = self.core.config_state();
        let identity_moved = (
            config.account_id.as_deref(),
            config.account_secret.as_deref(),
        ) != (
            self.config.account_id.as_deref(),
            self.config.account_secret.as_deref(),
        );
        self.config = config;
        if identity_moved {
            self.forget_account_scoped_state();
        }
        cx.notify();
    }

    fn forget_account_scoped_state(&mut self) {
        self.balances = Loadable::NotLoaded;
        self.balances_task = None;
        self.subscription = Loadable::NotLoaded;
        self.subscription_task = None;
        self.checkout_out = false;
    }

    /// The configured account is now a different one (created or linked):
    /// drop what the last one owned and read the new one's standing.
    /// Called on the commit, never on the attempt.
    pub fn identity_changed(&mut self, cx: &mut Context<Self>) {
        self.config = self.core.config_state();
        self.forget_account_scoped_state();
        self.refresh_balances(cx);
        self.refresh_subscription(cx);
        cx.notify();
    }

    pub fn refresh_balances(&mut self, cx: &mut Context<Self>) {
        if !self.has_account() {
            self.balances = Loadable::NotLoaded;
            return;
        }
        self.balances = std::mem::replace(&mut self.balances, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        self.balances_task = Some(cx.spawn(async move |this, cx| {
            let result = core.account_balances().await;
            this.update(cx, |this, cx| {
                this.balances_task = None;
                this.balances =
                    std::mem::replace(&mut this.balances, Loadable::NotLoaded).resolve(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    /// A verified account's balance, as the check that linked it read it.
    pub fn set_balances(&mut self, balances: BalancesResult, cx: &mut Context<Self>) {
        self.balances = Loadable::Loaded {
            value: balances,
            stale: false,
        };
        self.balances_task = None;
        cx.notify();
    }

    pub fn refresh_prices(&mut self, cx: &mut Context<Self>) {
        self.prices = std::mem::replace(&mut self.prices, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        self.prices_task = Some(cx.spawn(async move |this, cx| {
            let result = core.account_prices().await;
            this.update(cx, |this, cx| {
                this.prices_task = None;
                this.prices =
                    std::mem::replace(&mut this.prices, Loadable::NotLoaded).resolve(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    pub fn refresh_subscription(&mut self, cx: &mut Context<Self>) {
        if !self.has_account() {
            self.subscription = Loadable::NotLoaded;
            return;
        }
        self.subscription =
            std::mem::replace(&mut self.subscription, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        self.subscription_task = Some(cx.spawn(async move |this, cx| {
            let result = core.account_subscription().await;
            this.update(cx, |this, cx| {
                this.subscription_task = None;
                this.subscription =
                    std::mem::replace(&mut this.subscription, Loadable::NotLoaded).resolve(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    /// Re-read the documents the server requires acceptance of. The consent
    /// screen renders this snapshot, and creation submits the same value.
    pub fn refresh_terms(&mut self, cx: &mut Context<Self>) {
        self.terms = std::mem::replace(&mut self.terms, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        self.terms_task = Some(cx.spawn(async move |this, cx| {
            let result = core.current_terms().await;
            this.update(cx, |this, cx| {
                this.terms_task = None;
                this.terms =
                    std::mem::replace(&mut this.terms, Loadable::NotLoaded).resolve(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    pub fn refresh_backends(&mut self, cx: &mut Context<Self>) {
        self.eidola_enabled =
            std::mem::replace(&mut self.eidola_enabled, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        self.backends_task = Some(cx.spawn(async move |this, cx| {
            let result = core.list_backends().await.map(|backends| {
                backends
                    .iter()
                    .find(|b| b.id == EIDOLA_BACKEND)
                    .is_some_and(|b| b.enabled)
            });
            this.update(cx, |this, cx| {
                this.backends_task = None;
                this.eidola_enabled =
                    std::mem::replace(&mut this.eidola_enabled, Loadable::NotLoaded)
                        .resolve(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    /// Turn the Eidola backend on or off. Going on without an account turns
    /// it off (so the first run does not open again); setting up an account
    /// turns it back on.
    pub fn set_eidola_enabled(
        &self,
        enabled: bool,
    ) -> impl std::future::Future<Output = Result<(), AppError>> + 'static {
        let core = self.core.clone();
        async move {
            core.set_backend_enabled(EIDOLA_BACKEND.into(), enabled)
                .await
        }
    }

    /// Create an account, recording acceptance of `accepted_terms`: the
    /// snapshot this store's [`terms`](Self::terms) held when the reader
    /// agreed. The caller awaits it in its own slot.
    pub fn request_create(
        &self,
        accepted_terms: Vec<TermsDocument>,
    ) -> impl std::future::Future<Output = Result<AccountCreateResult, AppError>> + 'static {
        let core = self.core.clone();
        async move { core.account_create(accepted_terms).await }
    }

    /// Link an existing account: set its credentials, then read its balance.
    /// A refused check rolls the credentials back, so a bad attempt leaves
    /// the profile as it was.
    pub fn request_verify(
        &self,
        id: String,
        secret: String,
    ) -> impl std::future::Future<Output = Result<BalancesResult, AppError>> + 'static {
        let core = self.core.clone();
        async move {
            core.set_account_credentials(id, secret)?;
            match core.account_balances().await {
                Ok(balances) => Ok(balances),
                Err(e) => {
                    let _ = core.reset_account();
                    Err(e)
                }
            }
        }
    }

    /// Mint a checkout link for `price_id`, under the account configured
    /// when the request is signed ([`CheckoutMint::minted_for`]).
    pub fn request_checkout(
        &self,
        price_id: String,
    ) -> impl std::future::Future<Output = Result<CheckoutMint, AppError>> + 'static {
        let core = self.core.clone();
        async move { core.account_checkout(price_id).await }
    }

    /// Mint a billing-portal session at the moment of use (it is short-lived).
    pub fn request_portal(
        &self,
    ) -> impl std::future::Future<Output = Result<PortalMint, AppError>> + 'static {
        let core = self.core.clone();
        async move { core.account_portal().await }
    }

    /// Whether a link minted for `minted_for` belongs to the account this
    /// profile holds **now**: app-core's config, read past this store's
    /// snapshot, which lags it by a bus tick.
    pub fn mint_is_current(&self, minted_for: &str) -> bool {
        self.core.account_fingerprint().as_deref() == Some(minted_for)
    }

    /// A checkout link went to the browser.
    pub fn checkout_opened(&mut self) {
        self.checkout_out = true;
    }

    /// The app came back to the foreground: after a checkout, read the
    /// balance (and the subscription a purchase may have started) again.
    pub fn returned_to_foreground(&mut self, cx: &mut Context<Self>) {
        if std::mem::take(&mut self.checkout_out) {
            self.refresh_balances(cx);
            self.refresh_subscription(cx);
        }
    }

    /// Forget this profile's account credentials. The wallet's credentials
    /// stay (they are spendable without an account); the balance and the
    /// subscription go with the account they described.
    pub fn reset(&mut self, cx: &mut Context<Self>) {
        self.op_error = None;
        match self.core.reset_account() {
            Ok(()) => {
                self.config = self.core.config_state();
                self.forget_account_scoped_state();
            }
            Err(e) => self.op_error = Some(e),
        }
        cx.notify();
    }
}

/// The wallet's credentials and where each stands (read-only).
pub struct WalletStore {
    core: Rc<AppCore>,
    lifecycle: Loadable<Vec<CredentialLifecycleInfo>>,
    task: Option<Task<()>>,
}

impl WalletStore {
    pub fn new(core: Rc<AppCore>) -> Self {
        Self {
            core,
            lifecycle: Loadable::NotLoaded,
            task: None,
        }
    }

    pub fn lifecycle(&self) -> &Loadable<Vec<CredentialLifecycleInfo>> {
        &self.lifecycle
    }

    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        self.lifecycle = std::mem::replace(&mut self.lifecycle, Loadable::NotLoaded).to_loading();
        let core = self.core.clone();
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = core.wallet_lifecycle().await;
            this.update(cx, |this, cx| {
                this.task = None;
                this.lifecycle =
                    std::mem::replace(&mut this.lifecycle, Loadable::NotLoaded).resolve(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    /// Re-read only if something is on screen or was asked for.
    pub fn on_changed(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.lifecycle, Loadable::NotLoaded) {
            self.refresh(cx);
        }
    }
}
