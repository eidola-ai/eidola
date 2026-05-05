use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AccountCreateResult, AllocateResult, AppCore, BalancesResult, ChatResult, ChatStreamEvent,
    ConfigState, CredentialInfo, InFlightCredentialInfo, ModelInfo, PriceInfo, SpaceMessage,
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

    pub error_message: Option<String>,
    pub busy: bool,
}

impl Core {
    pub fn new(cx: &mut App) -> Entity<Self> {
        let config_dir = eidola_app_core::default_config_dir()
            .expect("could not determine eidola config directory");
        let data_dir =
            eidola_app_core::default_data_dir().expect("could not determine eidola data directory");

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

    fn refresh_config(&mut self, cx: &mut Context<Self>) {
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

    fn spawn<MakeFut, Fut, T, OnDone>(
        &mut self,
        cx: &mut Context<Self>,
        make_fut: MakeFut,
        on_done: OnDone,
    ) where
        MakeFut: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T, AppError>> + Send + 'static,
        T: Send + 'static,
        OnDone: FnOnce(&mut Core, Result<T, AppError>, &mut Context<Core>) + 'static,
    {
        if self.busy {
            return;
        }
        let Some(inner) = self.inner.as_ref() else {
            // Stub core (snapshot tests) — no backend to drive.
            return;
        };
        self.busy = true;
        self.error_message = None;
        cx.notify();

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
                this.busy = false;
                on_done(this, result, cx);
                cx.notify();
            });
        });
        task.detach();
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
