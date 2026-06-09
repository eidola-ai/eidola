use std::sync::Arc;
use std::time::Duration;

use eidola_app_core::error::AppError;
use eidola_app_core::{ChatStreamEvent, SpaceMessage};
use gpui::{
    AppContext, AsyncApp, Context, Div, Entity, EventEmitter, FocusHandle, Focusable,
    InteractiveElement, IntoElement, ParentElement, Render, SharedString,
    StatefulInteractiveElement, Styled, Subscription, WeakEntity, Window, actions, div,
    linear_color_stop, linear_gradient, px, relative, rems,
};
use gpui_component::{
    ActiveTheme, Disableable, IconName,
    button::{Button, ButtonVariants},
    h_flex,
    highlighter::HighlightTheme,
    label::Label,
    text::{TextView, TextViewStyle},
    v_flex,
};
use gpui_markdown_editor::{EditorState, MarkdownEditor, MarkdownStyle};

use crate::actions::CloseWindow;
use crate::core::Core;

/// Default model to send to the inference endpoint.
const DEFAULT_MODEL: &str = "gemma4-31b";

/// Vertical space reserved at the top of the window for the macOS traffic
/// lights. The window has a transparent titlebar (see
/// `lib.rs::transparent_titlebar`), so the OS draws the lights without
/// painting a separate titlebar background. This reserve does two things:
///
/// 1. Pads the messages list so the first message sits below the lights at
///    rest.
/// 2. Hosts a `theme.background → transparent` gradient overlay
///    (`render_title_bar_overlay`) painted on top of the scroll area, so
///    messages scrolling up under the band fade out smoothly into the chrome
///    instead of clipping at a hard edge.
#[cfg(target_os = "macos")]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(36.);
#[cfg(not(target_os = "macos"))]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(0.);

actions!(chat, [Send]);

/// In-flight assistant response. While this is `Some(...)`, the chat is
/// streaming — `reasoning` and `content` grow as deltas arrive. On
/// completion the streaming response is dropped; the captured reasoning
/// is moved onto the just-finalized assistant entry in `messages` so the
/// disclosure remains available after the stream ends.
#[derive(Default, Clone)]
pub struct StreamingResponse {
    pub reasoning: String,
    pub content: String,
    /// Whether the reasoning disclosure is open. Independent of whether
    /// reasoning has any content yet.
    pub expanded: bool,
    /// In-stream error: stream produced something the user should see,
    /// but the request as a whole has not necessarily failed.
    pub error: Option<String>,
}

/// A single rendered chat row: the persisted message plus any reasoning
/// captured for it during streaming. Reasoning is ephemeral session
/// state — the local DB stores only the assistant's final content — so
/// older messages from a re-loaded space carry `reasoning = None`. New
/// assistant messages adopt whatever reasoning was streaming at finalize.
#[derive(Clone)]
pub struct ChatMessageView {
    pub message: SpaceMessage,
    pub reasoning: Option<String>,
    pub reasoning_expanded: bool,
}

impl ChatMessageView {
    pub fn new(message: SpaceMessage) -> Self {
        Self {
            message,
            reasoning: None,
            reasoning_expanded: false,
        }
    }
}

/// The chat window's onboarding stage, derived each render from the shared
/// `Core` snapshots plus the view's local state. See
/// [`ChatView::onboarding_stage`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OnboardingStage {
    /// No account configured — the empty page is the welcome page.
    Welcome,
    /// Account exists, but the balance is known-zero and the wallet has no
    /// credentials — the empty page lists the plans.
    Plans,
    /// The normal page (blank or with transcript). Credential provisioning
    /// from the account balance is silent (app-core).
    Ready,
}

/// Local, per-window state for the onboarding flow. The *stage* is derived
/// (`ChatView::onboarding_stage`); every in-flight flag here corresponds to
/// a real request that is currently running or an explicit user choice —
/// no fake states.
#[derive(Default)]
pub struct OnboardingFlow {
    /// Account-creation request in flight.
    pub creating_account: bool,
    /// Error from the last account-creation attempt.
    pub create_error: Option<String>,
    /// `price_id` of a checkout-session request in flight.
    pub checkout_pending: Option<String>,
    /// A checkout URL has been opened in the browser and the balance poll
    /// is running.
    pub awaiting_checkout: bool,
    /// Error from the last checkout attempt or balance poll.
    pub plans_error: Option<String>,
    /// The user chose "I'll do this later" — fall through to the normal
    /// blank page for this window.
    pub dismissed: bool,
    /// Set when the account was just created from this window. Keeps the
    /// flow on the plans page while the first prices/balances fetch is in
    /// flight, instead of flashing the blank page between the two states.
    pub entered_plans: bool,
    /// A plans-data fetch has actually started (or the prices were already
    /// cached) for the current visit to the plans stage. Guards the
    /// observe-driven auto-fetch from looping. Deliberately *not* set when
    /// `Core::spawn`'s busy-debounce drops the call — the busy op's
    /// completion notify is the retry trigger.
    plans_fetch_attempted: bool,
    /// The 3s balance-poll loop is already running.
    poll_running: bool,
}

pub struct ChatView {
    core: Entity<Core>,
    /// WYSIWYG markdown composer (`gpui-markdown-editor`). The user types
    /// here in styled-markdown view; on submit we read `state.markdown` and
    /// send the raw source upstream.
    prompt_editor: Entity<MarkdownEditor>,
    space_id: Option<String>,
    /// Conversation history shown in the scroll view. `pub` so snapshot tests
    /// can render the view in a populated state without driving async chat.
    pub messages: Vec<ChatMessageView>,
    /// In-flight streaming assistant response, or `None` when idle.
    pub streaming: Option<StreamingResponse>,
    error: Option<String>,
    /// Onboarding flow state (welcome → plans → ready).
    onboarding: OnboardingFlow,
    /// A chat submit failed with `InsufficientBalance` — surface the plans
    /// list below the transcript's error band (not a modal).
    pub show_plans_after_error: bool,

    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl ChatView {
    /// The focus handle the view tracks. Exposed so behavior tests can dispatch
    /// actions through it the same way real keystrokes would.
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    /// Test-only access to the prompt editor entity, for behavior tests
    /// that want to populate it the way a typing user would (by writing
    /// `EditorState` directly).
    #[doc(hidden)]
    pub fn prompt_editor_for_test(&self) -> Entity<MarkdownEditor> {
        self.prompt_editor.clone()
    }

    /// Test-only setter for snapshot tests. Wraps each `SpaceMessage`
    /// into a `ChatMessageView` with `reasoning = None`.
    #[doc(hidden)]
    pub fn set_messages_for_test(&mut self, messages: Vec<SpaceMessage>) {
        self.messages = messages.into_iter().map(ChatMessageView::new).collect();
    }

    /// Test-only: attach reasoning to the message at `idx`, optionally
    /// expanded. Use after `set_messages_for_test`.
    #[doc(hidden)]
    pub fn set_reasoning_for_test(&mut self, idx: usize, reasoning: String, expanded: bool) {
        if let Some(entry) = self.messages.get_mut(idx) {
            entry.reasoning = Some(reasoning);
            entry.reasoning_expanded = expanded;
        }
    }

    /// Test-only setter for snapshot tests.
    #[doc(hidden)]
    pub fn set_streaming_for_test(&mut self, streaming: Option<StreamingResponse>) {
        self.streaming = streaming;
    }

    /// Read access to the onboarding flow state, for behavior tests.
    pub fn onboarding(&self) -> &OnboardingFlow {
        &self.onboarding
    }

    /// Test-only mutable access to the onboarding flow state, so snapshot
    /// tests can render in-flight sub-states (waiting for checkout, etc.)
    /// without driving async work.
    #[doc(hidden)]
    pub fn onboarding_mut_for_test(&mut self) -> &mut OnboardingFlow {
        &mut self.onboarding
    }

    /// Test-only setter for the transcript error band.
    #[doc(hidden)]
    pub fn set_error_for_test(&mut self, error: Option<String>) {
        self.error = error;
    }

    /// The space this window is writing into, if one has been assigned —
    /// either passed at construction (opened from the Library) or set when
    /// the first exchange creates a space.
    pub fn space_id(&self) -> Option<&str> {
        self.space_id.as_deref()
    }
}

impl ChatView {
    /// Construct a chat view. `space_id: None` is the blank page (⌘N): a
    /// fresh space is created lazily by the first exchange. `Some(id)`
    /// reopens an existing space (the Library's path): its messages are
    /// loaded asynchronously on construction and the next exchange
    /// continues that space.
    pub fn new(
        core: Entity<Core>,
        space_id: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        // The composer is a WYSIWYG markdown editor configured to match
        // the chat transcript's prose typography (Newsreader 17px / 1.65×
        // / gentle heading scale / 1.5 rem paragraph gap), so what the
        // user types renders the same way the assistant's reply will once
        // it lands in the transcript. The pixel-fidelity goal is spelled
        // out in `crates/gpui-markdown-editor/AGENTS.md`.
        let prompt_editor =
            cx.new(|cx| MarkdownEditor::new("", window, cx).style(composer_markdown_style(cx)));

        let focus_handle = cx.focus_handle();

        // Focus the composer so the user can start typing immediately,
        // like opening a fresh journal page. The view's `focus_handle` is
        // still tracked on the root v_flex (behavior tests dispatch
        // `Send` through it), but production focus lives on the editor
        // itself — which is the right cursor home for a "letter writing"
        // feel.
        let editor_focus = prompt_editor.read(cx).focus_handle(cx);
        window.focus(&editor_focus, cx);

        // ⌘↩ submits without a custom subscription dance: the editor's
        // `MarkdownEditor` key context does not bind `cmd-enter`, so the
        // ChatView-context `cmd-enter → Send` binding wins as the
        // innermost matching entry in the focus chain and dispatches
        // through to `Self::submit` on the v_flex root. Plain `enter`
        // (no modifier) is bound in the editor's context and inserts
        // a newline as normal.

        // One combined startup fetch: models, plus balances + wallet
        // credentials when an account exists (the onboarding inputs).
        core.update(cx, |core, cx| core.fetch_chat_startup(cx));

        // Re-render when the shared core snapshots change (config,
        // balances, prices, credentials), and lazily fetch the plans data
        // the first time the derived stage lands on `Plans`.
        let _subscriptions = vec![cx.observe(&core, |this: &mut Self, _, cx| {
            this.maybe_fetch_plans_data(cx);
            cx.notify();
        })];

        // Reopening an existing space: load its persisted messages in the
        // background, same bridge `spawn_stream` uses for its post-stream
        // re-fetch. Stub cores (tests) skip the load — tests preload via
        // `set_messages_for_test`.
        if let Some(sid) = space_id.clone()
            && let Some(app_core) = core.read(cx).app_core()
        {
            let msgs_rx = Core::get_space_messages(app_core, sid);
            cx.spawn_in(window, async move |this: WeakEntity<Self>, cx| {
                let msgs = msgs_rx.await.unwrap_or_else(|_| {
                    Err(eidola_app_core::error::AppError::Internal {
                        message: "fetch messages task cancelled".into(),
                    })
                });
                let _ = this.update(cx, |this, cx| {
                    match msgs {
                        Ok(messages) => this.merge_messages_from_db(messages, None),
                        Err(e) => this.error = Some(e.to_string()),
                    }
                    cx.notify();
                });
            })
            .detach();
        }

        Self {
            core,
            prompt_editor,
            space_id,
            messages: Vec::new(),
            streaming: None,
            error: None,
            onboarding: OnboardingFlow::default(),
            show_plans_after_error: false,
            focus_handle,
            _subscriptions,
        }
    }

    /// Derive the onboarding stage from the shared core snapshots and the
    /// view's local state.
    ///
    /// The onboarding pages only ever replace the *empty* page: any
    /// transcript content, in-flight stream, error band, or composer text
    /// short-circuits to `Ready` (a later funding failure is surfaced below
    /// the transcript via the error band instead — see
    /// `apply_chat_failure`).
    ///
    /// `Plans` requires the balance to be *known* zero (a fetched snapshot,
    /// not an assumption) with no wallet credentials. While the very first
    /// balances fetch after account creation is still in flight,
    /// `entered_plans` holds the flow on the plans page so it doesn't flash
    /// through the blank page.
    pub fn onboarding_stage(&self, core: &Core, composer_empty: bool) -> OnboardingStage {
        if !self.messages.is_empty()
            || self.streaming.is_some()
            || self.error.is_some()
            || !composer_empty
        {
            return OnboardingStage::Ready;
        }
        let Some(state) = core.config_state.as_ref() else {
            return OnboardingStage::Ready;
        };
        if !state.has_account || !state.has_account_secret {
            return OnboardingStage::Welcome;
        }
        if self.onboarding.dismissed {
            return OnboardingStage::Ready;
        }
        let balance_known_zero = core.balances.as_ref().is_some_and(|b| b.available <= 0);
        if balance_known_zero && core.credentials.is_empty() {
            return OnboardingStage::Plans;
        }
        if self.onboarding.entered_plans && core.balances.is_none() {
            return OnboardingStage::Plans;
        }
        OnboardingStage::Ready
    }

    fn current_stage(&self, cx: &Context<Self>) -> OnboardingStage {
        let composer_empty = self.prompt_editor.read(cx).state.markdown.trim().is_empty();
        self.onboarding_stage(self.core.read(cx), composer_empty)
    }

    /// Kick off a prices+balances fetch the first time the derived stage
    /// lands on `Plans` and there is nothing cached yet. One attempt per
    /// visit; the plans page's retry link clears the latch.
    fn maybe_fetch_plans_data(&mut self, cx: &mut Context<Self>) {
        if self.onboarding.plans_fetch_attempted {
            return;
        }
        if self.current_stage(cx) != OnboardingStage::Plans {
            return;
        }
        if !self.core.read(cx).prices.is_empty() {
            self.onboarding.plans_fetch_attempted = true;
            return;
        }
        // Latch only if the fetch actually started: `Core::spawn` silently
        // drops the call while another core op is in flight (e.g. the
        // startup fetch), and that op's completion notify is what re-runs
        // us for the retry.
        self.onboarding.plans_fetch_attempted =
            self.core.update(cx, |core, cx| core.fetch_plans_data(cx));
    }

    /// "Begin" on the welcome page: create the anonymous account. There is
    /// nothing to fill in — the account is an anonymous identifier.
    pub fn begin_onboarding(&mut self, cx: &mut Context<Self>) {
        if self.onboarding.creating_account {
            return;
        }
        self.onboarding.creating_account = true;
        self.onboarding.create_error = None;
        cx.notify();

        let Some(app_core) = self.core.read(cx).app_core() else {
            // Stub core (behavior tests): the in-flight flag above is the
            // observable state machine transition.
            return;
        };
        let rx = Core::account_create(app_core);
        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "account creation task cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                this.onboarding.creating_account = false;
                match res {
                    Ok(_) => {
                        // Move straight to the plans step: refresh the
                        // config snapshot (has_account flips true) and
                        // fetch prices + the initial (zero) balance. The
                        // latch tracks whether the fetch actually started —
                        // if the startup fetch is still busy the call is
                        // dropped, and `maybe_fetch_plans_data` retries on
                        // that op's completion notify.
                        this.onboarding.entered_plans = true;
                        this.onboarding.plans_fetch_attempted = this.core.update(cx, |core, cx| {
                            core.refresh_config(cx);
                            core.fetch_plans_data(cx)
                        });
                    }
                    Err(e) => this.onboarding.create_error = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// A plan row was clicked: create a checkout session, open the URL in
    /// the browser, and start polling balances until the purchase lands.
    pub fn begin_checkout(&mut self, price_id: String, cx: &mut Context<Self>) {
        if self.onboarding.checkout_pending.is_some() {
            return;
        }
        self.onboarding.checkout_pending = Some(price_id.clone());
        self.onboarding.plans_error = None;
        cx.notify();

        let Some(app_core) = self.core.read(cx).app_core() else {
            return;
        };
        let rx = Core::account_checkout(app_core, price_id);
        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "checkout task cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                this.onboarding.checkout_pending = None;
                match res {
                    Ok(url) => {
                        cx.open_url(&url);
                        this.onboarding.awaiting_checkout = true;
                        this.start_balance_poll(cx);
                    }
                    Err(e) => this.onboarding.plans_error = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Quiet exit from the plans page ("I'll do this later"): show the
    /// normal blank page and stop the checkout poll.
    pub fn dismiss_onboarding(&mut self, cx: &mut Context<Self>) {
        self.onboarding.dismissed = true;
        self.onboarding.awaiting_checkout = false;
        cx.notify();
    }

    /// Poll balances every ~3s while `awaiting_checkout`. Each iteration is
    /// a real `GET /v1/account/balances`; a positive balance ends both the
    /// poll and (via the derived stage) the plans page. Poll errors are
    /// surfaced inline and polling continues — checkout may still complete.
    fn start_balance_poll(&mut self, cx: &mut Context<Self>) {
        if self.onboarding.poll_running {
            return;
        }
        let Some(app_core) = self.core.read(cx).app_core() else {
            return;
        };
        self.onboarding.poll_running = true;
        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            loop {
                cx.background_executor().timer(Duration::from_secs(3)).await;
                let keep_going = this
                    .read_with(cx, |this, _| this.onboarding.awaiting_checkout)
                    .unwrap_or(false);
                if !keep_going {
                    break;
                }
                let res = match Core::account_balances(app_core.clone()).await {
                    Ok(res) => res,
                    Err(_) => break, // core dropped
                };
                let stop = this
                    .update(cx, |this, cx| {
                        match res {
                            Ok(balances) => {
                                this.onboarding.plans_error = None;
                                if balances.available > 0 {
                                    this.onboarding.awaiting_checkout = false;
                                }
                                this.core.update(cx, |core, cx| {
                                    core.balances = Some(balances);
                                    cx.notify();
                                });
                            }
                            Err(e) => {
                                this.onboarding.plans_error =
                                    Some(format!("balance check failed: {e}"));
                            }
                        }
                        cx.notify();
                        !this.onboarding.awaiting_checkout
                    })
                    .unwrap_or(true);
                if stop {
                    break;
                }
            }
            let _ = this.update(cx, |this, _| {
                this.onboarding.poll_running = false;
            });
        })
        .detach();
    }

    /// Route a failed chat submit. `InsufficientBalance` additionally
    /// surfaces the plans list below the transcript via the error band —
    /// typed routing, not string matching.
    pub fn apply_chat_failure(&mut self, e: AppError, cx: &mut Context<Self>) {
        self.streaming = None;
        if matches!(e, AppError::InsufficientBalance { .. }) {
            self.show_plans_after_error = true;
            if self.core.read(cx).prices.is_empty() {
                self.core.update(cx, |core, cx| core.fetch_plans_data(cx));
            }
        }
        self.error = Some(e.to_string());
        cx.notify();
    }

    /// Replace `messages` with a fresh list from the DB, preserving any
    /// previously-attached reasoning by index (we only ever append in
    /// this view, so positions are stable) and attaching the just-
    /// captured streaming reasoning to the new last assistant entry if
    /// non-empty.
    fn merge_messages_from_db(
        &mut self,
        new_messages: Vec<SpaceMessage>,
        new_reasoning: Option<String>,
    ) {
        let mut next: Vec<ChatMessageView> = new_messages
            .into_iter()
            .enumerate()
            .map(|(idx, msg)| {
                let prior = self.messages.get(idx);
                let same_position = prior.is_some_and(|p| {
                    p.message.role == msg.role && p.message.content == msg.content
                });
                ChatMessageView {
                    message: msg,
                    reasoning: if same_position {
                        prior.and_then(|p| p.reasoning.clone())
                    } else {
                        None
                    },
                    reasoning_expanded: if same_position {
                        prior.is_some_and(|p| p.reasoning_expanded)
                    } else {
                        false
                    },
                }
            })
            .collect();

        if let Some(reasoning) = new_reasoning
            && !reasoning.is_empty()
        {
            // Find the last assistant entry and attach the reasoning we
            // captured during streaming.
            if let Some(entry) = next
                .iter_mut()
                .rev()
                .find(|e| e.message.role == "assistant")
            {
                entry.reasoning = Some(reasoning);
            }
        }
        self.messages = next;
    }

    fn submit(&mut self, _: &Send, window: &mut Window, cx: &mut Context<Self>) {
        if self.streaming.is_some() {
            return;
        }

        let prompt = self
            .prompt_editor
            .read(cx)
            .state
            .markdown
            .trim()
            .to_string();
        if prompt.is_empty() {
            return;
        }

        self.prompt_editor.update(cx, |editor, cx| {
            editor.state = EditorState::default();
            cx.notify();
        });
        self.messages.push(ChatMessageView::new(SpaceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        }));
        self.streaming = Some(StreamingResponse::default());
        self.error = None;
        self.show_plans_after_error = false;
        cx.notify();

        let Some(app_core) = self.core.read(cx).app_core() else {
            // Stub core (behavior tests): the local state update above has
            // already happened; without a real backend there is nothing more
            // to do.
            return;
        };
        let space_id = self.space_id.clone();

        self.spawn_stream(app_core, prompt, space_id, window, cx);
    }

    /// Drive a streaming chat request from gpui's main thread.
    fn spawn_stream(
        &mut self,
        app_core: Arc<eidola_app_core::AppCore>,
        prompt: String,
        space_id: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (mut event_rx, done_rx) =
            Core::chat_stream(app_core.clone(), prompt, DEFAULT_MODEL.into(), space_id);

        cx.spawn_in(window, async move |this: WeakEntity<Self>, cx| {
            while let Some(event) = event_rx.recv().await {
                let _ = this.update(cx, |this, cx| {
                    if let Some(s) = this.streaming.as_mut() {
                        match event {
                            ChatStreamEvent::ReasoningDelta(d) => s.reasoning.push_str(&d),
                            ChatStreamEvent::ContentDelta(d) => s.content.push_str(&d),
                        }
                    }
                    cx.notify();
                });
            }

            let outcome = done_rx.await.unwrap_or_else(|_| {
                Err(eidola_app_core::error::AppError::Internal {
                    message: "chat task cancelled".into(),
                })
            });

            match outcome {
                Ok(result) => {
                    let msgs_rx = Core::get_space_messages(app_core, result.space_id.clone());
                    let msgs = msgs_rx.await.unwrap_or_else(|_| {
                        Err(eidola_app_core::error::AppError::Internal {
                            message: "fetch messages task cancelled".into(),
                        })
                    });
                    let _ = this.update(cx, |this, cx| {
                        let captured_reasoning =
                            this.streaming.as_ref().map(|s| s.reasoning.clone());
                        this.streaming = None;
                        this.space_id = Some(result.space_id);
                        match msgs {
                            Ok(messages) => {
                                this.merge_messages_from_db(messages, captured_reasoning)
                            }
                            Err(e) => this.error = Some(e.to_string()),
                        }
                        cx.notify();
                    });
                }
                Err(e) => {
                    let _ = this.update(cx, |this, cx| {
                        this.apply_chat_failure(e, cx);
                    });
                }
            }
        })
        .detach();
    }
}

impl EventEmitter<()> for ChatView {}

impl Render for ChatView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // The onboarding pages are the chat window's *empty states* — the
        // welcome page when no account exists, the plans page when the
        // account is unfunded. Any real page content (or composer text)
        // short-circuits to the normal transcript.
        let stage = self.current_stage(cx);
        let content: gpui::AnyElement = match stage {
            OnboardingStage::Welcome => self.render_welcome(window, cx).into_any_element(),
            OnboardingStage::Plans => self.render_plans_page(window, cx).into_any_element(),
            OnboardingStage::Ready => self.render_transcript(window, cx).into_any_element(),
        };

        let theme = cx.theme();
        v_flex()
            .key_context("ChatView")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::submit))
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .relative()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(content)
            .child(title_bar_overlay(cx))
    }
}

impl ChatView {
    /// The normal page: transcript + composer in one scroll surface.
    fn render_transcript(&self, window: &Window, cx: &Context<Self>) -> gpui::Stateful<Div> {
        let theme = cx.theme();
        let markdown_style = markdown_style(theme.mode.is_dark());

        // The composer carries a bottom padding of half the viewport
        // height (computed here, applied below) so the cursor never sits
        // pinned to the bottom edge of the window. As the user types and
        // the cursor moves down, the parent scroll follows; the half-page
        // of empty space below keeps the active line in the comfortable
        // reading zone — the "writing surface scrolls under your hand"
        // feel of long-form note tools.
        let composer_pb = window.viewport_size().height * 0.5;

        // pt(TITLE_BAR_RESERVE) handles the leading edge under the
        // traffic lights. No trailing pb on the column itself: the
        // composer's own pb (above) carries the bottom breath.
        let mut messages_col = v_flex().w_full().gap_0().pt(TITLE_BAR_RESERVE);
        for (idx, entry) in self.messages.iter().enumerate() {
            let msg = &entry.message;

            // Chapter delimiter — a hairline rule + italic participant
            // name — replaces the per-row background tinting that used to
            // distinguish speakers. Errors still use `theme.danger` for the
            // label so the chrome itself signals the role.
            //
            // The very first message in a conversation has no leading
            // delim: the user's text is always the *start* of the page
            // (no header introducing it), and the second turn's delim
            // (e.g. "Eidola") is what first signals a speaker change.
            // Subsequent same-speaker messages still get their delim so
            // the rhythm is preserved across turns.
            if idx > 0 {
                let label_color = if msg.role == "error" {
                    theme.danger
                } else {
                    theme.muted_foreground
                };
                messages_col = messages_col.child(chapter_delim(
                    participant_label(&msg.role),
                    theme.border,
                    label_color,
                ));
            }

            let fg = if msg.role == "error" {
                theme.danger
            } else {
                theme.foreground
            };

            let mut row = v_flex()
                .id(("msg-row", idx))
                .w_full()
                .px_5()
                .gap_3()
                .text_color(fg);

            if let Some(reasoning) = entry.reasoning.as_deref()
                && msg.role == "assistant"
            {
                let chevron = if entry.reasoning_expanded {
                    IconName::ChevronDown
                } else {
                    IconName::ChevronRight
                };
                // Wrap the Button in an `h_flex` so it sizes to its
                // content; without a flex-row parent the button
                // stretches to v_flex's full cross-axis width and the
                // label ends up center-aligned. Centered inside a
                // `prose_row` so the disclosure aligns with the reading
                // column underneath rather than the row's hard left edge.
                row = row.child(
                    prose_row().child(
                        prose().child(
                            h_flex().child(
                                Button::new(("toggle-thinking", idx))
                                    .ghost()
                                    .icon(chevron)
                                    .label(SharedString::from("Thinking"))
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        if let Some(entry) = this.messages.get_mut(idx) {
                                            entry.reasoning_expanded = !entry.reasoning_expanded;
                                            cx.notify();
                                        }
                                    })),
                            ),
                        ),
                    ),
                );
                if entry.reasoning_expanded {
                    row = row.child(
                        prose_row().child(
                            prose().pl_4().text_color(theme.muted_foreground).child(
                                TextView::markdown(("thinking-body", idx), reasoning.to_string())
                                    .selectable(true)
                                    .style(markdown_style.clone()),
                            ),
                        ),
                    );
                }
            }

            let body: gpui::AnyElement = if msg.role == "error" {
                SharedString::from(msg.content.clone()).into_any_element()
            } else {
                TextView::markdown(("msg", idx), msg.content.clone())
                    .selectable(true)
                    .style(markdown_style.clone())
                    .into_any_element()
            };
            row = row.child(prose_row().child(prose().child(body)));

            messages_col = messages_col.child(row);
        }

        if let Some(s) = self.streaming.as_ref() {
            let fg = theme.foreground;
            let muted_fg = theme.muted_foreground;
            let danger = theme.danger;

            // Same chapter delimiter pattern as a finalized assistant
            // message; the row underneath will fill in as deltas arrive.
            messages_col = messages_col.child(chapter_delim(
                participant_label("assistant"),
                theme.border,
                muted_fg,
            ));

            let mut col = v_flex()
                .id("streaming-row")
                .w_full()
                .px_5()
                .gap_3()
                .text_color(fg);

            // Disclosure only appears once reasoning has actually
            // arrived; before that we just show a "Thinking…" status so
            // the user sees something is in flight.
            if !s.reasoning.is_empty() {
                let chevron = if s.expanded {
                    IconName::ChevronDown
                } else {
                    IconName::ChevronRight
                };
                col = col.child(
                    prose_row().child(
                        prose().child(
                            h_flex().child(
                                Button::new("toggle-thinking-streaming")
                                    .ghost()
                                    .icon(chevron)
                                    .label(SharedString::from("Thinking"))
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        if let Some(s) = this.streaming.as_mut() {
                                            s.expanded = !s.expanded;
                                            cx.notify();
                                        }
                                    })),
                            ),
                        ),
                    ),
                );
                if s.expanded {
                    col = col.child(
                        prose_row().child(
                            prose().pl_4().text_color(muted_fg).child(
                                TextView::markdown(
                                    ("streaming-thinking-body", 0usize),
                                    s.reasoning.clone(),
                                )
                                .selectable(true)
                                .style(markdown_style.clone()),
                            ),
                        ),
                    );
                }
            } else if s.content.is_empty() {
                // No reasoning *and* no content yet — show the "still
                // working" status indicator. Plain Label, no toggle, no
                // markdown plumbing. Aligned with the prose column so the
                // status doesn't visually jump when content arrives.
                col = col.child(prose_row().child(
                    prose().child(Label::new(SharedString::from("Thinking…")).text_color(muted_fg)),
                ));
            }

            if !s.content.is_empty() {
                col = col.child(
                    prose_row().child(
                        prose().child(
                            TextView::markdown(("streaming-body", 0usize), s.content.clone())
                                .selectable(true)
                                .style(markdown_style.clone()),
                        ),
                    ),
                );
            }

            if let Some(err) = s.error.as_deref() {
                col = col.child(
                    prose_row().child(
                        prose().text_color(danger).child(
                            TextView::markdown(("streaming-error", 0usize), err.to_string())
                                .selectable(true)
                                .style(markdown_style.clone()),
                        ),
                    ),
                );
            }

            messages_col = messages_col.child(col);
        }

        if let Some(err) = self.error.as_ref() {
            // Errors are rendered through `TextView::markdown` instead
            // of a raw `SharedString` so the user can select and copy
            // the text. `selectable(true)` is the only thing here that's
            // load-bearing for that — markdown of a plain string is just
            // plain text. The chapter delim's "Error" label is in
            // `theme.danger`, and the body inherits the danger color
            // through the row's `text_color`.
            messages_col = messages_col.child(chapter_delim(
                participant_label("error"),
                theme.border,
                theme.danger,
            ));
            messages_col = messages_col.child(
                div().w_full().px_5().text_color(theme.danger).child(
                    prose_row().child(
                        prose().child(
                            TextView::markdown(("chat-error", 0usize), err.clone())
                                .selectable(true)
                                .style(markdown_style.clone()),
                        ),
                    ),
                ),
            );

            // A submit that failed with `InsufficientBalance` surfaces the
            // plans right here, below the transcript — the same hairline
            // list the onboarding plans page uses, not a modal.
            if self.show_plans_after_error {
                messages_col = messages_col.child(
                    div().w_full().px_5().pt_6().child(
                        prose_row().child(
                            prose_col()
                                .gap_4()
                                .child(
                                    div()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child("Add credit to continue"),
                                )
                                .child(self.render_plans_list(cx)),
                        ),
                    ),
                );
            }
        }

        // The composing editor lives at the foot of the scroll, styled
        // into the same prose column the body uses so it reads as a
        // continuation of the page rather than a separate chrome
        // element. A "You" chapter delim sits above it whenever there's
        // preceding content, mirroring the way earlier turns are
        // introduced; on a fresh, empty page the delim is omitted so the
        // cursor sits cleanly at the top.
        //
        // The editor renders one gpui block per markdown block, so it
        // grows naturally with content. The outer `overflow_y_scroll`
        // div handles all of it (editor + preceding messages) as one
        // continuous unit — the editor itself does not scroll
        // internally.
        //
        // `min_h(...)` keeps the empty editor clickable even when its
        // markdown is "" (the render pipeline emits no blocks in that
        // state, so without a floor the editor would collapse to zero
        // height and the user couldn't click back into it after losing
        // focus). The floor matches one body line at the prose line-
        // height ratio so it doesn't visually grow once content arrives.
        let has_preceding =
            !self.messages.is_empty() || self.streaming.is_some() || self.error.is_some();
        if has_preceding {
            messages_col = messages_col.child(chapter_delim(
                participant_label("user"),
                theme.border,
                theme.muted_foreground,
            ));
        }
        let composer_min_h = PROSE_FONT_SIZE * PROSE_LINE_HEIGHT;
        messages_col = messages_col.child(
            div().w_full().px_5().pb(composer_pb).child(
                prose_row().child(
                    prose()
                        .min_h(composer_min_h)
                        .child(self.prompt_editor.clone()),
                ),
            ),
        );

        div()
            .id("scroll")
            .w_full()
            .flex_1()
            .overflow_y_scroll()
            .child(messages_col)
    }

    /// The welcome page — the empty state when no account exists. Set like
    /// a title page in the prose column: the wordmark, a short hairline,
    /// three sentences of what Eidola is, and a single action.
    fn render_welcome(&self, window: &Window, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let ob = &self.onboarding;

        let wordmark = v_flex()
            .gap_4()
            .child(
                div()
                    .text_size(px(34.))
                    .line_height(relative(1.2))
                    .child("Eidola"),
            )
            .child(div().w(rems(4.)).h(px(1.)).bg(theme.border));

        // gap_6 = 1.5 rem — the same paragraph gap the transcript's
        // markdown body uses, so the welcome reads with the page's rhythm.
        let body = v_flex()
            .gap_6()
            .child(div().child(
                "A quiet page for thinking with a machine — private by construction, \
                 not by policy.",
            ))
            .child(div().child(
                "Every request runs inside sealed, hardware-attested enclaves, and this \
                 app verifies the cryptographic evidence before a word leaves your \
                 machine. The full record of what was sent, spent, and proven lives \
                 here, on your device.",
            ))
            .child(div().child(
                "There is nothing to sign up for: an account is just an anonymous \
                 number. Press Begin, and the page is yours.",
            ));

        let mut action = v_flex().gap_3().items_start().child(
            h_flex().child(
                Button::new("begin")
                    .primary()
                    .label("Begin")
                    .disabled(ob.creating_account)
                    .on_click(cx.listener(|this, _, _, cx| this.begin_onboarding(cx))),
            ),
        );
        if ob.creating_account {
            // Real in-flight request — see `begin_onboarding`.
            action = action.child(
                div()
                    .text_sm()
                    .italic()
                    .text_color(theme.muted_foreground)
                    .child("Creating your anonymous account…"),
            );
        }
        if let Some(err) = ob.create_error.as_deref() {
            action = action.child(
                div()
                    .text_sm()
                    .text_color(theme.danger)
                    .child(SharedString::from(err.to_string())),
            );
        }

        v_flex()
            .size_full()
            .pt(window.viewport_size().height * 0.22)
            .px_5()
            .child(
                prose_row().child(
                    prose_col()
                        .gap_8()
                        .child(wordmark)
                        .child(body)
                        .child(action),
                ),
            )
    }

    /// The plans page — the empty state when the account exists but holds
    /// no balance and the wallet has no credentials.
    fn render_plans_page(&self, window: &Window, cx: &Context<Self>) -> Div {
        let theme = cx.theme();

        let heading = v_flex()
            .gap_3()
            .child(
                div()
                    .text_size(px(22.))
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child("Add credit"),
            )
            .child(div().text_color(theme.muted_foreground).child(
                "Your account is ready. Choose a plan to put credit behind it — \
                 checkout opens in your browser, and this page follows along.",
            ));

        let skip = h_flex().pt_2().child(
            div()
                .id("skip-plans")
                .cursor_pointer()
                .text_sm()
                .text_color(theme.muted_foreground)
                .hover(|s| s.text_color(theme.foreground))
                .child("I'll do this later")
                .on_click(cx.listener(|this, _, _, cx| this.dismiss_onboarding(cx))),
        );

        v_flex()
            .size_full()
            .pt(window.viewport_size().height * 0.16)
            .px_5()
            .child(
                prose_row().child(
                    prose_col()
                        .gap_6()
                        .child(heading)
                        .child(self.render_plans_list(cx))
                        .child(skip),
                ),
            )
    }

    /// The plans list itself — shared between the onboarding plans page and
    /// the below-transcript "add credit to continue" band. Hairline rules,
    /// no cards: each plan is a row (name · price; credits underneath), and
    /// clicking it opens checkout.
    fn render_plans_list(&self, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let core = self.core.read(cx);
        let ob = &self.onboarding;

        let mut list = v_flex().w_full();

        if core.prices.is_empty() {
            if core.busy {
                // `fetch_plans_data` is in flight.
                list = list.child(
                    div()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .child("Loading plans…"),
                );
            } else if let Some(err) = core.error_message.as_deref() {
                list = list
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.danger)
                            .child(SharedString::from(err.to_string())),
                    )
                    .child(
                        div()
                            .id("retry-plans")
                            .cursor_pointer()
                            .text_sm()
                            .text_color(theme.muted_foreground)
                            .hover(|s| s.text_color(theme.foreground))
                            .child("Try again")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.onboarding.plans_fetch_attempted =
                                    this.core.update(cx, |core, cx| {
                                        core.clear_error(cx);
                                        core.fetch_plans_data(cx)
                                    });
                            })),
                    );
            } else {
                list = list.child(
                    div()
                        .text_color(theme.muted_foreground)
                        .child("No plans are available right now."),
                );
            }
            return list;
        }

        for (idx, price) in core.prices.iter().enumerate() {
            let price_line = if ob.checkout_pending.as_deref() == Some(price.id.as_str()) {
                // Real in-flight request — see `begin_checkout`.
                "Opening checkout…".to_string()
            } else {
                format!("{}{}", price.amount_display, price.recurrence)
            };
            let mut subline = format!("{} credits", format_credits(price.credits));
            if let Some(desc) = price.product_description.as_deref() {
                subline = format!("{subline} — {desc}");
            }
            let price_id = price.id.clone();

            list =
                list.child(
                    v_flex()
                        .id(("plan", idx))
                        .w_full()
                        .py_3()
                        .gap_1()
                        .border_t_1()
                        .border_color(theme.border)
                        .cursor_pointer()
                        .hover(|s| s.bg(theme.muted.opacity(0.35)))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.begin_checkout(price_id.clone(), cx)
                        }))
                        .child(
                            h_flex()
                                .w_full()
                                .justify_between()
                                .items_baseline()
                                .child(div().child(SharedString::from(price.product_name.clone())))
                                .child(
                                    div()
                                        .text_color(theme.muted_foreground)
                                        .child(SharedString::from(price_line)),
                                ),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.muted_foreground)
                                .child(SharedString::from(subline)),
                        ),
                );
        }
        // Closing hairline under the last row.
        list = list.child(div().w_full().h(px(1.)).bg(theme.border));

        if ob.awaiting_checkout {
            // The balance poll is live — one real request every ~3s.
            list = list.child(
                div()
                    .pt_4()
                    .italic()
                    .text_color(theme.muted_foreground)
                    .child(
                        "Waiting for checkout to finish in your browser — checking your \
                         balance every few seconds…",
                    ),
            );
        }
        if let Some(err) = ob.plans_error.as_deref() {
            list = list.child(
                div()
                    .pt_2()
                    .text_sm()
                    .text_color(theme.danger)
                    .child(SharedString::from(err.to_string())),
            );
        }

        list
    }
}

/// Body font size for chat content. We intentionally bump above the 16px UI
/// baseline: Newsreader at 17px sits in proportion with a hardcopy book page,
/// where the body text is the dominant visual element rather than a UI label.
const PROSE_FONT_SIZE: gpui::Pixels = gpui::px(17.);

/// Line-height as a fraction of font size (CSS-style unitless leading). 1.65×
/// is the readability sweet spot for serifs at this size — generous enough
/// that descenders and ascenders never crowd each other, tight enough that the
/// eye still tracks paragraph cohesion.
const PROSE_LINE_HEIGHT: f32 = 1.65;

/// Maximum width of the reading column, in `rem` units (anchored to the
/// theme's base 16px, so this is ~640px). At ~9.5px average glyph width for
/// Newsreader at 17px, that lands around 65–72 characters per line — the
/// canonical measure for long-form readability. The column is centered in
/// the row so wide windows don't force the eye to track across the screen.
const PROSE_MAX_WIDTH_REM: f32 = 40.;

/// Wrap a single block of book-typography content. Used around every
/// `TextView::markdown` invocation in the chat (message bodies, reasoning
/// disclosures, streaming partials, errors) so they all share one reading
/// column. The wrapper is the *single* place body size, leading, and measure
/// are set; the markdown renderer then inherits them through gpui's normal
/// text-style cascade.
///
/// We do not center via `mx_auto`: v_flex children stretch to full width by
/// default, so the wrapper stays full-width and the centering is done one
/// level up via `prose_row()`.
fn prose() -> Div {
    div()
        .w_full()
        .max_w(rems(PROSE_MAX_WIDTH_REM))
        .text_size(PROSE_FONT_SIZE)
        .line_height(relative(PROSE_LINE_HEIGHT))
}

/// Center the prose column horizontally within a full-width row. Wraps the
/// content in an `h_flex` with `justify_center` so the inner div with
/// `max_w` lands centered in wide windows.
fn prose_row() -> Div {
    h_flex().w_full().justify_center()
}

/// `prose()` as a vertical flex column, for multi-child onboarding content
/// (the plain `prose()` div is block-level, where `gap` has no effect).
fn prose_col() -> Div {
    v_flex()
        .w_full()
        .max_w(rems(PROSE_MAX_WIDTH_REM))
        .text_size(PROSE_FONT_SIZE)
        .line_height(relative(PROSE_LINE_HEIGHT))
}

/// Format a credit amount with thousands separators (credits are micro-USD
/// denominated, so the magnitudes are large).
fn format_credits(credits: i64) -> String {
    let raw = credits.abs().to_string();
    let mut out = String::with_capacity(raw.len() + raw.len() / 3 + 1);
    if credits < 0 {
        out.push('-');
    }
    let offset = raw.len() % 3;
    for (i, ch) in raw.chars().enumerate() {
        if i > 0 && (i + 3 - offset).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Friendly label for a chat role, used as the participant name in the
/// chapter-delimiter rule between turns.
fn participant_label(role: &str) -> &'static str {
    match role {
        "user" => "You",
        "assistant" => "Eidola",
        "error" => "Error",
        _ => "—",
    }
}

/// A book-style "chapter" delimiter between message turns: a hairline rule
/// running across the prose column, broken in the middle by a small italic
/// label naming the upcoming participant. This replaces the alternating
/// per-message backgrounds we used previously — a real book doesn't tint
/// each speaker's paragraphs, it sets them apart with whitespace and a
/// rule.
///
/// Layout strategy (mirrors `gpui_component::Divider`'s horizontal-with-
/// label pattern):
///
/// - The outer frame is the same `.px_5() → prose_row() → prose()` chain
///   the message bodies use, so the delim's reading column tracks the
///   body's reading column.
/// - Inside the prose column, a `relative` flex container holds two
///   children: an **absolute-positioned hairline** that spans the full
///   inset width (`w_full + h(1px)`), and a **centered label** painted on
///   top with `bg(bg_color)` that masks the rule directly behind it.
///   `items_center + justify_center` align the label both axes; the
///   absolute hairline is out of flex flow.
///
/// Earlier iterations used `flex_1` rule divs on either side of the
/// label. That works when the container has a definite main-axis size,
/// but inside a nested `prose() → h_flex()` chain the flex item sizing
/// rules collapsed the rules to zero width when `prose`'s `max_w` bound,
/// leaving a label alone with no rule. The absolute approach sidesteps
/// flex item sizing entirely — the rule is a non-flex child sized by
/// `w_full`, which always takes 100% of its (block-level) parent.
///
/// `rule_color` is normally `theme.border`; `label_color` is
/// `theme.muted_foreground` for normal turns and `theme.danger` for the
/// error band so the chrome itself signals the role. `bg_color` must
/// match the surrounding chat background so the label cleanly masks the
/// rule beneath it.
/// A book-style "chapter" delimiter between message turns: a hairline rule
/// across the prose column, broken in the middle by a small italic label
/// naming the upcoming participant. Replaces the alternating per-message
/// backgrounds we used previously — a real book doesn't tint each
/// speaker's paragraphs, it sets them apart with whitespace and a rule.
///
/// Layout: a centered `h_flex` capped at the prose column width, with two
/// `flex_1` hairline rules flanking the label. Whether this stretches
/// correctly depends on the *scroll container* upstream having `w_full`
/// — without it, `flex_1 + overflow_y_scroll` content-sizes the entire
/// downstream tree and the rules collapse along with it. See `Render`.
///
/// `rule_color` is normally `theme.border`; `label_color` is
/// `theme.muted_foreground` for normal turns and `theme.danger` for the
/// error band so the chrome itself signals the role.
fn chapter_delim(
    label: impl Into<SharedString>,
    rule_color: gpui::Hsla,
    label_color: gpui::Hsla,
) -> impl IntoElement {
    h_flex().w_full().justify_center().pt_8().pb_6().child(
        h_flex()
            .w_full()
            .max_w(rems(PROSE_MAX_WIDTH_REM))
            .items_center()
            .gap_4()
            .px_5()
            .child(div().h(px(1.)).flex_1().bg(rule_color))
            .child(
                div()
                    .text_sm()
                    .italic()
                    .text_color(label_color)
                    .child(label.into()),
            )
            .child(div().h(px(1.)).flex_1().bg(rule_color)),
    )
}

/// `MarkdownStyle` for the WYSIWYG composer. Shares the prose typography
/// the transcript uses (Newsreader 17px / 1.65× / gentle heading scale /
/// 1.5 rem paragraph gap) so what the user types reads pixel-for-pixel
/// like the assistant's reply will once it lands above as a finalized
/// message. Theme-derived colors (text, delimiter, background, caret,
/// selection) are refreshed inside the editor's `Render`, so we only
/// need to override the typography knobs here.
fn composer_markdown_style(cx: &gpui::App) -> MarkdownStyle {
    MarkdownStyle::from_theme(cx)
        .font_size(PROSE_FONT_SIZE)
        .line_height(rems(PROSE_LINE_HEIGHT))
        .paragraph_gap(rems(1.5))
        .heading_base_font_size(PROSE_FONT_SIZE)
        .heading_font_size(|level, base| match level {
            1 => base * 1.5,
            2 => base * 1.25,
            3 => base * 1.125,
            _ => base,
        })
}

/// `TextViewStyle` for chat message bodies. Settings:
///
/// - `is_dark` and `highlight_theme` track the active Circadian mode so
///   fenced code blocks render against the right backdrop.
/// - `heading_base_font_size` is anchored to the body size (instead of the
///   default 14px) so the rem-based scale below is interpreted relative to
///   what the reader is actually reading.
/// - The heading-size callback applies a gentler scale than the gpui default
///   (which jumps h1 to 2× and h2 to 1.5× of a 14px base, pushing h1 toward
///   marketing-site weight). A book has a flatter type ramp; weight carries
///   most of the hierarchy and size only widens at the top of the heading
///   tree. Returned scale: h1 1.5× / h2 1.25× / h3 1.125× / h4-6 1.0×.
fn markdown_style(is_dark: bool) -> TextViewStyle {
    let highlight = if is_dark {
        HighlightTheme::default_dark().clone()
    } else {
        HighlightTheme::default_light().clone()
    };
    TextViewStyle {
        is_dark,
        highlight_theme: highlight,
        heading_base_font_size: PROSE_FONT_SIZE,
        // Paragraph gap of 1.5 rem (~24px at the 16px theme rem) — about
        // 85% of a body line. Books either run paragraphs flush with a
        // first-line indent or break them with a clear half-to-full line
        // of breath; the renderer doesn't expose first-line indent, so we
        // lean fully into spacing. The gpui default of 1.0 rem read as a
        // chat tool's tight stack rather than a book page's clear break.
        paragraph_gap: rems(1.5),
        ..TextViewStyle::default()
    }
    .heading_font_size(|level, base| match level {
        1 => base * 1.5,
        2 => base * 1.25,
        3 => base * 1.125,
        _ => base,
    })
}

/// Title-bar overlay: a gradient that fades from full `theme.background` at
/// the top to fully transparent at the bottom of the reserve. Painted over
/// the scroll area (positioned absolutely, last child wins z-order in gpui),
/// so messages scrolling up under it dissolve smoothly instead of clipping.
///
/// Two non-aesthetic modifiers tame the title-bar band:
///
/// - `.cursor_default()` sets the platform cursor to `Arrow` over the band
///   *and* causes gpui to register a hitbox here
///   (`Interactivity::should_insert_hitbox` includes
///   `style.mouse_cursor.is_some()`). Without it, the text below keeps
///   winning the cursor-style lookup and the I-beam shows over the band.
/// - `.block_mouse_except_scroll()` upgrades that hitbox to swallow click
///   and drag events so a double-click-drag in the band doesn't fall
///   through to `TextView`'s selectable handler and start a text
///   selection underneath. Scroll passes through, so wheel-scrolling
///   while the cursor is in the band still scrolls the chat.
///
/// macOS native titlebar behavior (drag, double-click-to-zoom) is handled
/// by AppKit at the NSWindow layer before the gpui content view is asked,
/// so blocking mouse on the gpui side doesn't disturb it.
fn title_bar_overlay(cx: &gpui::App) -> impl IntoElement {
    let bg = cx.theme().background;
    div()
        .absolute()
        .top_0()
        .left_0()
        .right_0()
        .h(TITLE_BAR_RESERVE)
        .cursor_default()
        .block_mouse_except_scroll()
        .bg(linear_gradient(
            180.,
            linear_color_stop(bg, 0.0),
            linear_color_stop(bg.opacity(0.0), 1.0),
        ))
}
