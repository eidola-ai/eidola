use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::updates::UpdateCheckSnapshot;
use eidola_app_core::{
    AccountCreateResult, AllocateResult, AppCore, BalancesResult, ChatResult, ChatStreamEvent,
    ConfigState, CredentialInfo, InFlightCredentialInfo, ModelInfo, PriceInfo, SpaceInfo,
    SpaceMessage,
};
use gpui::{App, AppContext, AsyncApp, Context, Entity, Task, WeakEntity};
use tokio::sync::{mpsc, oneshot};

/// Reactive wrapper around `AppCore`.
///
/// Owns an `Arc<AppCore>` so async tasks can be spawned on the core's tokio
/// runtime and still share the same instance across views. Cached state
/// (config snapshot, balances, prices, …) is stored here so any view holding
/// `Entity<Core>` can read the latest value via `.read(cx)` and re-render on
/// `cx.notify()`.
pub struct Core {
    /// Real backend. `None` only in snapshot/visual tests, where views are
    /// rendered against a fixed cached state and never trigger async work.
    inner: Option<Arc<AppCore>>,

    pub config_state: Option<ConfigState>,
    pub balances: Option<BalancesResult>,
    pub prices: Vec<PriceInfo>,
    pub credentials: Vec<CredentialInfo>,
    pub spending_credentials: Vec<InFlightCredentialInfo>,
    pub models: Vec<ModelInfo>,
    /// Cached space listing for the Library window (archived excluded).
    pub spaces: Vec<SpaceInfo>,
    /// Outcome of the most recent completed update check (manual or
    /// background poll). The Updates window reads this reactively.
    pub update_check: Option<UpdateCheckSnapshot>,
    /// True while a manual update check is in flight.
    pub update_checking: bool,

    pub error_message: Option<String>,
    pub busy: bool,
}

impl Core {
    pub fn new(cx: &mut App) -> Entity<Self> {
        let config_dir = eidola_app_core::config::default_config_path()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .expect("could not determine eidola config directory");
        let data_dir = eidola_app_core::config::default_data_dir()
            .expect("could not determine eidola data directory");

        let inner = Arc::new(AppCore::new(config_dir, data_dir));
        let config_state = Some(inner.config_state());

        cx.new(|_| Self {
            inner: Some(inner),
            config_state,
            balances: None,
            prices: Vec::new(),
            credentials: Vec::new(),
            spending_credentials: Vec::new(),
            models: Vec::new(),
            spaces: Vec::new(),
            update_check: None,
            update_checking: false,
            error_message: None,
            busy: false,
        })
    }

    /// Builds a `Core` with no real backend, for use in snapshot tests.
    /// Tests mutate the public state fields directly to set up the scene to
    /// render. Calling any method that needs the backend will panic.
    pub fn stub() -> Self {
        Self {
            inner: None,
            config_state: None,
            balances: None,
            prices: Vec::new(),
            credentials: Vec::new(),
            spending_credentials: Vec::new(),
            models: Vec::new(),
            spaces: Vec::new(),
            update_check: None,
            update_checking: false,
            error_message: None,
            busy: false,
        }
    }

    /// Direct access to the underlying `AppCore`. Use this when a view needs
    /// to spawn its own async work (e.g. chat completion) on the core's
    /// runtime without going through a cached field on `Core`. Returns `None`
    /// for stub cores (snapshot/behavior tests); callers should treat that as
    /// "do the local state update but skip any backend work".
    pub fn app_core(&self) -> Option<Arc<AppCore>> {
        self.inner.clone()
    }

    /// Re-read the config snapshot from the backend. Public so views that
    /// drive config-mutating operations directly on `AppCore` (e.g. the
    /// chat onboarding flow's account creation) can refresh the shared
    /// snapshot afterwards.
    pub fn refresh_config(&mut self, cx: &mut Context<Self>) {
        if let Some(inner) = self.inner.as_ref() {
            self.config_state = Some(inner.config_state());
            cx.notify();
        }
    }

    fn set_error(&mut self, err: AppError, cx: &mut Context<Self>) {
        self.error_message = Some(err.to_string());
        cx.notify();
    }

    #[allow(dead_code)]
    pub fn clear_error(&mut self, cx: &mut Context<Self>) {
        if self.error_message.take().is_some() {
            cx.notify();
        }
    }

    // ------------------------------------------------------------------
    // Config — synchronous mutations
    // ------------------------------------------------------------------

    pub fn set_base_url(&mut self, url: String, cx: &mut Context<Self>) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        match inner.set_base_url(url) {
            Ok(()) => self.refresh_config(cx),
            Err(e) => self.set_error(e, cx),
        }
    }

    /// Persist the user's default model (`default_model` config override)
    /// and refresh the shared config snapshot so every window's resolved
    /// default tracks the new value. No-op on stub cores: behavior tests
    /// assert per-window selection state instead, and the config round-trip
    /// is covered at the app-core layer.
    pub fn set_default_model(&mut self, model: String, cx: &mut Context<Self>) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        match inner.set_default_model(model) {
            Ok(()) => self.refresh_config(cx),
            Err(e) => self.set_error(e, cx),
        }
    }

    #[allow(dead_code)]
    pub fn set_attestation_url(&mut self, url: String, cx: &mut Context<Self>) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        match inner.set_attestation_url(url) {
            Ok(()) => self.refresh_config(cx),
            Err(e) => self.set_error(e, cx),
        }
    }

    #[allow(dead_code)]
    pub fn set_account_credentials(&mut self, id: String, secret: String, cx: &mut Context<Self>) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        match inner.set_account_credentials(id, secret) {
            Ok(()) => self.refresh_config(cx),
            Err(e) => self.set_error(e, cx),
        }
    }

    pub fn reset_account(&mut self, cx: &mut Context<Self>) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        match inner.reset_account() {
            Ok(()) => self.refresh_config(cx),
            Err(e) => self.set_error(e, cx),
        }
    }

    // ------------------------------------------------------------------
    // Account — async operations
    // ------------------------------------------------------------------

    pub fn create_account(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn(
            cx,
            move || async move { core.account_create().await },
            |this, result, cx| match result {
                Ok(_) => this.refresh_config(cx),
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    pub fn fetch_balances(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn(
            cx,
            move || async move { core.account_balances().await },
            |this, result, cx| match result {
                Ok(b) => {
                    this.balances = Some(b);
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    pub fn fetch_prices(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn(
            cx,
            move || async move { core.account_prices().await },
            |this, result, cx| match result {
                Ok(p) => {
                    this.prices = p;
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    pub fn fetch_credentials(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn(
            cx,
            move || async move {
                let active = core.wallet_credentials().await?;
                let spending = core.wallet_spending_credentials().await?;
                Ok((active, spending))
            },
            |this, result, cx| match result {
                Ok((active, spending)) => {
                    this.credentials = active;
                    this.spending_credentials = spending;
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    /// Attempt recovery of any in-flight credentials. After the call
    /// returns, refreshes both active and spending credentials caches and
    /// stashes the recovered nonces into `last_recovered` so the UI can
    /// surface a result message. The result count is delivered via the
    /// `on_done` callback so callers can show an alert.
    pub fn recover_spending_credentials(
        &mut self,
        cx: &mut Context<Self>,
        on_done: impl FnOnce(&mut Core, Vec<String>, &mut Context<Core>) + 'static,
    ) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn(
            cx,
            move || {
                let core1 = core.clone();
                async move {
                    let recovered = core1.recover_spending_credentials().await?;
                    let active = core1.wallet_credentials().await?;
                    let spending = core1.wallet_spending_credentials().await?;
                    Ok((recovered, active, spending))
                }
            },
            move |this, result, cx| match result {
                Ok((recovered, active, spending)) => {
                    this.credentials = active;
                    this.spending_credentials = spending;
                    cx.notify();
                    on_done(this, recovered, cx);
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    pub fn fetch_models(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn(
            cx,
            move || async move { core.available_models().await },
            |this, result, cx| match result {
                Ok(m) => {
                    this.models = m;
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    /// One-shot startup fetch for a chat window: the model list, plus —
    /// when an account is configured — balances and wallet credentials
    /// (the inputs the onboarding state machine needs to decide between
    /// the plans page and the normal blank page).
    ///
    /// Combined into a single `spawn` because `Core::spawn` debounces on
    /// `busy`: separate sequential calls would silently drop all but the
    /// first.
    pub fn fetch_chat_startup(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        let has_account = self
            .config_state
            .as_ref()
            .is_some_and(|s| s.has_account && s.has_account_secret);
        self.spawn(
            cx,
            move || async move {
                let models = core.available_models().await?;
                let account = if has_account {
                    let balances = core.account_balances().await?;
                    let credentials = core.wallet_credentials().await?;
                    Some((balances, credentials))
                } else {
                    None
                };
                Ok((models, account))
            },
            |this, result, cx| match result {
                Ok((models, account)) => {
                    this.models = models;
                    if let Some((balances, credentials)) = account {
                        this.balances = Some(balances);
                        this.credentials = credentials;
                    }
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    /// Refresh the cached space listing (archived spaces excluded).
    ///
    /// Uses `spawn_unguarded` rather than `spawn`: the shared `busy` flag
    /// exists so settings/account views can disable their buttons during
    /// an operation, but the Library must be able to refresh even while
    /// some other core call (e.g. a new chat window's model fetch) is in
    /// flight — `spawn`'s busy gate would silently drop the refresh.
    pub fn fetch_spaces(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn_unguarded(
            cx,
            move || async move { core.list_spaces(false).await },
            |this, result, cx| match result {
                Ok(s) => {
                    this.spaces = s;
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    /// Fetch the data the onboarding plans page needs — prices and a fresh
    /// balance snapshot — in a single `spawn` (see `fetch_chat_startup` for
    /// why they're combined).
    ///
    /// Returns whether the fetch actually started (see `spawn`): callers
    /// latch `plans_fetch_attempted` on this, so a busy-debounced drop stays
    /// retryable on the next core notification.
    pub fn fetch_plans_data(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(core) = self.inner.clone() else {
            return false;
        };
        self.spawn(
            cx,
            move || async move {
                let prices = core.account_prices().await?;
                let balances = core.account_balances().await?;
                Ok((prices, balances))
            },
            |this, result, cx| match result {
                Ok((prices, balances)) => {
                    this.prices = prices;
                    this.balances = Some(balances);
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        )
    }

    /// Archive a space. The cached row is removed immediately (so the
    /// Library updates without waiting on the backend — and so stub-core
    /// tests can exercise the local path); the backend archive then runs
    /// and the listing is re-fetched to reconcile. Unguarded for the same
    /// reason as `fetch_spaces`, but more so: a busy-gated drop here would
    /// leave the row removed locally while the space was never archived.
    pub fn archive_space(&mut self, space_id: String, cx: &mut Context<Self>) {
        self.spaces.retain(|s| s.id != space_id);
        cx.notify();

        let Some(core) = self.inner.clone() else {
            return;
        };
        self.spawn_unguarded(
            cx,
            move || async move {
                core.archive_space(space_id).await?;
                core.list_spaces(false).await
            },
            |this, result, cx| match result {
                Ok(s) => {
                    this.spaces = s;
                    cx.notify();
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    // ------------------------------------------------------------------
    // Updates — verified update-notification flow
    // ------------------------------------------------------------------

    /// Pull the persisted last-check snapshot into the cache (covers
    /// background-poll results landing while no Updates window was open —
    /// the next window open reflects them).
    pub fn load_last_update_check(&mut self, cx: &mut Context<Self>) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.update_check = inner.last_update_check();
        cx.notify();
    }

    /// Run a manual update check. Unguarded (must never be silently
    /// dropped by the shared `busy` debounce); its own `update_checking`
    /// flag makes re-entry a no-op while one is in flight.
    pub fn check_for_updates(&mut self, cx: &mut Context<Self>) {
        if self.update_checking {
            return;
        }
        let Some(core) = self.inner.clone() else {
            return;
        };
        self.update_checking = true;
        cx.notify();
        self.spawn_unguarded(
            cx,
            move || async move { Ok(core.update_check().await) },
            |this, result, cx| {
                this.update_checking = false;
                if let Ok(snapshot) = result {
                    this.update_check = Some(snapshot);
                }
                cx.notify();
            },
        );
    }

    /// Record the user's explicit "treat as update" decision for a
    /// claims-changed release, then refresh the cached snapshot so the
    /// window re-renders as an available update.
    pub fn accept_changed_claims(
        &mut self,
        version: String,
        manifest_sha256: String,
        cx: &mut Context<Self>,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        match inner.accept_changed_claims(version, manifest_sha256) {
            Ok(()) => {
                self.update_check = inner.last_update_check();
                cx.notify();
            }
            Err(e) => self.set_error(e, cx),
        }
    }

    /// Start the app-core background poll loop (launch + every ~6h).
    /// Idempotent; no-op on stub cores.
    pub fn start_update_polling(&self) {
        if let Some(inner) = self.inner.as_ref() {
            inner.start_update_polling();
        }
    }

    pub fn allocate_credits(&mut self, credits: i64, cx: &mut Context<Self>) {
        let Some(core) = self.inner.clone() else {
            return;
        };
        let core_for_then = core.clone();
        self.spawn(
            cx,
            move || async move { core.account_allocate(credits).await },
            move |this, result, cx| match result {
                Ok(_) => {
                    // Refresh credentials and balances after a successful allocation.
                    let core1 = core_for_then.clone();
                    let core2 = core_for_then.clone();
                    this.spawn(
                        cx,
                        move || async move {
                            let active = core1.wallet_credentials().await?;
                            let spending = core1.wallet_spending_credentials().await?;
                            Ok((active, spending))
                        },
                        |this, result, cx| {
                            if let Ok((active, spending)) = result {
                                this.credentials = active;
                                this.spending_credentials = spending;
                                cx.notify();
                            }
                        },
                    );
                    this.spawn(
                        cx,
                        move || async move { core2.account_balances().await },
                        |this, result, cx| {
                            if let Ok(b) = result {
                                this.balances = Some(b);
                                cx.notify();
                            }
                        },
                    );
                }
                Err(e) => this.set_error(e, cx),
            },
        );
    }

    // ------------------------------------------------------------------
    // Internal: bridge a tokio future into the gpui main-thread context.
    // ------------------------------------------------------------------

    /// Returns whether the work was actually spawned: `false` when debounced
    /// by `busy` or on a stub core. Callers that latch "I already fetched"
    /// state must only set the latch on `true` — a dropped call completes no
    /// future and updates no cache, so latching it would wedge the UI on
    /// stale emptiness (the core notifies when the busy op finishes, which is
    /// the retry signal).
    fn spawn<MakeFut, Fut, T, OnDone>(
        &mut self,
        cx: &mut Context<Self>,
        make_fut: MakeFut,
        on_done: OnDone,
    ) -> bool
    where
        MakeFut: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T, AppError>> + Send + 'static,
        T: Send + 'static,
        OnDone: FnOnce(&mut Core, Result<T, AppError>, &mut Context<Core>) + 'static,
    {
        if self.busy {
            return false;
        }
        if self.inner.is_none() {
            // Stub core (snapshot tests) — no backend to drive. Bail before
            // touching `busy` so it never sticks on.
            return false;
        }
        self.busy = true;
        self.error_message = None;
        cx.notify();

        self.spawn_unguarded(cx, make_fut, |this, result, cx| {
            this.busy = false;
            on_done(this, result, cx);
        });
        true
    }

    /// Like `spawn`, but neither checks nor sets the shared `busy` flag.
    /// For operations that must not be silently dropped when something
    /// else is in flight (space listing refresh, archive).
    fn spawn_unguarded<MakeFut, Fut, T, OnDone>(
        &mut self,
        cx: &mut Context<Self>,
        make_fut: MakeFut,
        on_done: OnDone,
    ) -> bool
    where
        MakeFut: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T, AppError>> + Send + 'static,
        T: Send + 'static,
        OnDone: FnOnce(&mut Core, Result<T, AppError>, &mut Context<Core>) + 'static,
    {
        let Some(inner) = self.inner.as_ref() else {
            // Stub core (snapshot tests) — no backend to drive.
            return false;
        };

        let handle = inner.runtime().handle().clone();
        let (tx, rx) = oneshot::channel();
        handle.spawn(async move {
            let res = make_fut().await;
            let _ = tx.send(res);
        });

        let task: Task<()> = cx.spawn(async move |this: WeakEntity<Core>, cx: &mut AsyncApp| {
            let result = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "background task cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                on_done(this, result, cx);
                cx.notify();
            });
        });
        task.detach();
        true
    }

    // ------------------------------------------------------------------
    // Direct chat helpers — used by ChatView, which manages its own state.
    // ------------------------------------------------------------------

    pub fn chat(
        core: Arc<AppCore>,
        prompt: String,
        model: String,
        space_id: Option<String>,
    ) -> oneshot::Receiver<Result<ChatResult, AppError>> {
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let res = core.chat(prompt, model, space_id).await;
            let _ = tx.send(res);
        });
        rx
    }

    /// Create an anonymous account on the server. Used by the chat
    /// window's onboarding welcome page; unlike the `create_account`
    /// entity method, this returns the typed result so the caller owns
    /// its own in-flight/error state and is not subject to the `busy`
    /// debounce.
    pub fn account_create(
        core: Arc<AppCore>,
    ) -> oneshot::Receiver<Result<AccountCreateResult, AppError>> {
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let res = core.account_create().await;
            let _ = tx.send(res);
        });
        rx
    }

    /// Create a checkout session for `price_id` and return the URL to open
    /// in the browser.
    pub fn account_checkout(
        core: Arc<AppCore>,
        price_id: String,
    ) -> oneshot::Receiver<Result<String, AppError>> {
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let res = core.account_checkout(price_id).await;
            let _ = tx.send(res);
        });
        rx
    }

    /// Fetch the current balances. Used by the onboarding checkout poll.
    pub fn account_balances(
        core: Arc<AppCore>,
    ) -> oneshot::Receiver<Result<BalancesResult, AppError>> {
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let res = core.account_balances().await;
            let _ = tx.send(res);
        });
        rx
    }

    pub fn get_space_messages(
        core: Arc<AppCore>,
        space_id: String,
    ) -> oneshot::Receiver<Result<Vec<SpaceMessage>, AppError>> {
        let (tx, rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let res = core.get_space_messages(space_id).await;
            let _ = tx.send(res);
        });
        rx
    }

    /// Streaming chat. Spawns the streaming chat call on the core's tokio
    /// runtime and returns:
    ///
    /// - an `UnboundedReceiver<ChatStreamEvent>` for incremental
    ///   reasoning/content deltas (closes when the stream finishes), and
    /// - a `oneshot::Receiver<Result<ChatResult, AppError>>` for the
    ///   terminal outcome.
    ///
    /// The two are drained from gpui's main-thread context (see
    /// `chat::ChatView::submit_streaming`).
    pub fn chat_stream(
        core: Arc<AppCore>,
        prompt: String,
        model: String,
        space_id: Option<String>,
    ) -> (
        mpsc::UnboundedReceiver<ChatStreamEvent>,
        oneshot::Receiver<Result<ChatResult, AppError>>,
    ) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (done_tx, done_rx) = oneshot::channel();
        core.runtime().handle().clone().spawn(async move {
            let res = core.chat_stream(prompt, model, space_id, event_tx).await;
            let _ = done_tx.send(res);
        });
        (event_rx, done_rx)
    }
}

/// Helper to keep the result types accessible from view modules without re-exports.
#[allow(dead_code)]
pub type AccountCreateOutcome = Result<AccountCreateResult, AppError>;
#[allow(dead_code)]
pub type AllocateOutcome = Result<AllocateResult, AppError>;
