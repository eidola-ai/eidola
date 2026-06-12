use std::time::Duration;

use crate::window_input::WindowInput;
use eidola_app_core::error::AppError;
use eidola_app_core::{ModelInfo, SpaceMessage};
use gpui::{
    AnyElement, AppContext, AsyncApp, Context, Div, Entity, EventEmitter, FocusHandle, Focusable,
    InteractiveElement, IntoElement, ListAlignment, ListState, ModifiersChangedEvent,
    ParentElement, Render, ScrollHandle, SharedString, StatefulInteractiveElement, Styled,
    Subscription, WeakEntity, Window, actions, div, linear_color_stop, linear_gradient, list,
    prelude::FluentBuilder, px, relative, rems,
};
use gpui_component::{
    ActiveTheme, Disableable, IconName, StyledExt,
    button::{Button, ButtonVariants},
    h_flex,
    highlighter::HighlightTheme,
    label::Label,
    text::{TextView, TextViewStyle},
    v_flex,
};
use gpui_markdown_editor::{EditorState, MarkdownEditor, MarkdownStyle};

use crate::actions::CloseWindow;
use crate::plans::format_credits;
// Re-exported for the chat surface's consumers (tests construct
// `StreamingResponse` / read `ChatMessageView` through `eidola_gui::chat`).
pub use crate::space::{ChatMessageView, StreamingResponse};
use crate::space::{Space, SpaceEvent};
use crate::stores::{AccountStore, ConfigStore, ModelsStore, Stores, WalletStore};

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

/// Left clearance for the title-bar band's left-aligned participant indicator
/// so it doesn't render *behind* the macOS traffic lights (which AppKit draws
/// at the top-left of the transparent titlebar). Same clearance constant family
/// as `record::STRIP_LEFT_PAD` and gpui-component's `TITLE_BAR_LEFT_PADDING`
/// (80px). Platform-gated like the other traffic-light reserves — no pad is
/// needed off macOS, where the window has no overlaid stoplights.
#[cfg(target_os = "macos")]
const INDICATOR_LEFT_PAD: gpui::Pixels = gpui::px(80.);
#[cfg(not(target_os = "macos"))]
const INDICATOR_LEFT_PAD: gpui::Pixels = gpui::px(0.);

actions!(
    chat,
    [
        /// Submit the composer's markdown to the model. Bound to ⌘↩ in the
        /// `ChatView` key context.
        Send,
        /// Toggle the quiet model picker anchored to the title-bar band.
        /// Bound to ⌥⌘M in the `ChatView` key context; clicking the
        /// ⌥-revealed model label is the pointer path to the same state.
        ToggleModelPicker,
        /// Dismiss the model picker (Esc, while the picker is open).
        DismissModelPicker,
        /// Move the picker highlight one row up.
        PickerUp,
        /// Move the picker highlight one row down.
        PickerDown,
        /// Select the currently highlighted picker row (Enter).
        PickerConfirm,
    ]
);

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
    /// A plans-data fetch has started (or the prices were already cached) for
    /// the current visit to the plans stage. Guards the observe-driven
    /// auto-fetch from looping. Each store refresh owns its own task slot, so
    /// the fetch always starts — the latch is purely re-entry protection.
    plans_fetch_attempted: bool,
    /// The 3s balance-poll loop is already running.
    poll_running: bool,
}

pub struct ChatView {
    /// The store bundle — held whole because the chat surface reads several
    /// domains (config, models, account, wallet) and opens spaces through the
    /// `SpacesStore` registry.
    stores: Stores,
    config: Entity<ConfigStore>,
    models: Entity<ModelsStore>,
    account: Entity<AccountStore>,
    wallet: Entity<WalletStore>,
    /// The shared per-conversation domain entity. Owns the transcript,
    /// streaming buffers, the submit runner, and the per-space model
    /// selection — two windows on the same space hold the *same* `Space`, so
    /// a submit/stream in one appears in the other (wave-2 bug 4). `ChatView`
    /// is a window-local lens over it.
    space: Entity<Space>,
    /// WYSIWYG markdown composer (`gpui-markdown-editor`). The user types
    /// here in styled-markdown view; on submit we read `state.markdown` and
    /// send the raw source upstream. The composer draft is **window-local by
    /// design** — two windows on one space are two cursors with two drafts
    /// (see `docs/architecture/state.md`).
    prompt_editor: Entity<MarkdownEditor>,
    /// The error band shown below the transcript. Window-local: set from the
    /// space's `SpaceEvent::Failed` (and a transcript-load failure), so each
    /// window can surface its own last-submit error and its own degraded
    /// onboarding state.
    error: Option<String>,
    /// Onboarding flow state (welcome → plans → ready).
    onboarding: OnboardingFlow,
    /// A chat submit failed with `InsufficientBalance` — surface the plans
    /// list below the transcript's error band (not a modal).
    pub show_plans_after_error: bool,

    /// Per-window modifier state. The root registers the window's single
    /// `on_modifiers_changed` listener and mirrors every event here.
    /// `ChatView` observes this entity for the ⌥-reveal and picker-anchor
    /// behavior; it never registers its own modifier listener.
    window_input: Entity<WindowInput>,
    /// Whether the model picker panel is open (⌥⌘M or clicking the
    /// revealed label). The label stays revealed while the picker is open
    /// so the picker keeps its visual anchor after ⌥ is released.
    model_picker_open: bool,
    /// The index of the keyboard-highlighted row in the model picker, if
    /// any. `None` when the picker is closed or no row has been touched
    /// yet. Arrow keys move it; Enter selects; Esc dismisses.
    picker_highlighted: Option<usize>,
    /// Tracked scroll handle for the model picker's row container. The picker
    /// panel is a capped-height `overflow_y_scroll` div, so a long model list
    /// scrolls internally; this handle lets keyboard navigation
    /// (`PickerUp`/`PickerDown`, and opening with a far-down current selection)
    /// scroll the highlighted row into view via `scroll_to_item`.
    picker_scroll: ScrollHandle,
    /// Test-only mirror of the last row index handed to
    /// `picker_scroll.scroll_to_item` — `ScrollHandle` exposes no getter for its
    /// pending active item, so this lets behavior tests assert the picker
    /// scrolls to follow the keyboard highlight without a real paint pass.
    last_picker_scroll_target: Option<usize>,

    /// Virtualized-transcript scroll state. `list()` renders only the visible
    /// window of `transcript_items`, so per-frame work is O(visible). The
    /// element-side `ListState` is the persistent scroll handle (the old
    /// `overflow_y_scroll` div had none); `splice`/`remeasure`/`scroll_to_end`
    /// drive the tail policy (see `rebuild_transcript`). Window-local: each
    /// window has its own scroll position even on a shared `Space`.
    list_state: ListState,
    /// The current flat transcript model — one entry per `list()` item, the
    /// composer included as the final item. Recomputed by `rebuild_transcript`
    /// whenever the message/streaming/error shape changes; the render closure
    /// is a dumb indexer over it (the Record/Library idiom). Its `len()` is the
    /// list's item count.
    transcript_items: Vec<TranscriptItem>,
    /// The participant label shown in the title-bar band's left side while the
    /// top of the viewport sits inside a message whose chapter delim has
    /// scrolled off (see [`Self::participant_indicator`]). `None` at the top
    /// of the page or while a delim is visible. Recomputed from `list()` scroll
    /// state on every scroll and render — no layout shift (absolute chrome).
    participant_indicator: Option<ParticipantIndicator>,
    /// Test-only record of how the last *reshaping* `rebuild_transcript`
    /// reconciled the `ListState`: `Some(true)` = wholesale `reset`,
    /// `Some(false)` = provable tail-append fast-path. A no-op reconcile (model
    /// unchanged) leaves this untouched, so a re-render-only notify firing right
    /// after a real reshape doesn't erase the record. The regression replay
    /// asserts that a suffix-shifting reshape (first/second submit) `reset`s
    /// rather than mis-splicing a bogus tail.
    last_reconcile_was_reset: Option<bool>,

    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

/// One row of the virtualized transcript — a `list()` item. The chapter
/// delimiter that introduces a turn travels *with* its message item (rather
/// than being its own item) so the delim and the body it labels are measured
/// and scrolled as one unit, and a delim never lands alone on an item where
/// it could collapse. The composer is the final item, keeping the
/// messages-plus-composer single-scroll-surface feel under virtualization.
#[derive(Clone, Debug, PartialEq, Eq)]
enum TranscriptItem {
    /// A persisted transcript message at this index. Carries its own leading
    /// chapter delim unless it is the very first row of the page.
    Message { index: usize, leading_delim: bool },
    /// The in-flight streaming assistant row (delim + thinking + partial body).
    Streaming,
    /// The window-local error band (delim + body + optional plans list).
    Error,
    /// The composer item (leading "You" delim when there's preceding content,
    /// the editor, its min-height floor and half-viewport bottom padding).
    Composer { leading_delim: bool },
}

/// The resolved participant indicator: the delim-voice label plus whether it
/// is the error voice (which keeps the danger color). A pure projection of the
/// `list()` scroll state + the transcript model — see
/// [`ChatView::derive_participant_indicator`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantIndicator {
    /// The label text ("You" / "Eidola" / "Error").
    pub label: &'static str,
    /// The voice is an error turn — render in `theme.danger`, not muted.
    pub is_error: bool,
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

    /// The shared `Space` entity this window is a lens over.
    pub fn space(&self) -> &Entity<Space> {
        &self.space
    }

    /// The transcript rows, read from the shared `Space`.
    pub fn messages(&self, cx: &gpui::App) -> Vec<ChatMessageView> {
        self.space.read(cx).messages().to_vec()
    }

    /// The current streaming response, read from the shared `Space`.
    pub fn streaming(&self, cx: &gpui::App) -> Option<StreamingResponse> {
        self.space.read(cx).streaming().cloned()
    }

    /// Test-only setter for snapshot tests. Writes the transcript into the
    /// shared `Space`.
    #[doc(hidden)]
    pub fn set_messages_for_test(&mut self, messages: Vec<SpaceMessage>, cx: &mut Context<Self>) {
        self.space
            .update(cx, |s, cx| s.set_messages_for_test(messages, cx));
    }

    /// Test-only: attach reasoning to the message at `idx`, optionally
    /// expanded. Use after `set_messages_for_test`.
    #[doc(hidden)]
    pub fn set_reasoning_for_test(
        &mut self,
        idx: usize,
        reasoning: String,
        expanded: bool,
        cx: &mut Context<Self>,
    ) {
        self.space.update(cx, |s, cx| {
            s.set_reasoning_for_test(idx, reasoning, expanded, cx)
        });
    }

    /// Test-only setter for snapshot tests. Writes the streaming state into
    /// the shared `Space`.
    #[doc(hidden)]
    pub fn set_streaming_for_test(
        &mut self,
        streaming: Option<StreamingResponse>,
        cx: &mut Context<Self>,
    ) {
        self.space
            .update(cx, |s, cx| s.set_streaming_for_test(streaming, cx));
    }

    /// Test-only: push a streaming content delta into the shared `Space`,
    /// exactly as the real runner does (emits `SpaceEvent::StreamDelta`). The
    /// regression repro uses this to drive the "model started responding"
    /// moment that the disappearing-message bug surfaced at.
    #[doc(hidden)]
    pub fn push_content_delta_for_test(&mut self, delta: &str, cx: &mut Context<Self>) {
        self.space
            .update(cx, |s, cx| s.push_content_delta_for_test(delta, cx));
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
    /// the first exchange creates a space. Reads through the shared `Space`.
    pub fn space_id(&self, cx: &gpui::App) -> Option<String> {
        self.space.read(cx).id().map(str::to_string)
    }

    // -- Model picker state -------------------------------------------------

    /// The model the next send will use: the space's explicit selection if the
    /// user picked one, otherwise the config default
    /// (`ConfigState::default_model`), otherwise the embedded fallback
    /// (stub cores in tests have no config snapshot).
    pub fn current_model(&self, cx: &gpui::App) -> String {
        if let Some(model) = self.space.read(cx).selected_model() {
            return model.to_string();
        }
        self.config
            .read(cx)
            .state()
            .map(|s| s.default_model.clone())
            .unwrap_or_else(|| eidola_app_core::config::DEFAULT_MODEL.to_string())
    }

    /// Whether the model label is currently revealed in the title-bar band:
    /// while ⌥ is held (via `WindowInput`), or while the picker it anchors
    /// is open.
    pub fn model_revealed(&self, cx: &gpui::App) -> bool {
        self.window_input.read(cx).alt_held() || self.model_picker_open
    }

    /// Whether the model picker panel is open.
    pub fn model_picker_open(&self) -> bool {
        self.model_picker_open
    }

    /// The space's explicit model selection, if any.
    pub fn selected_model(&self, cx: &gpui::App) -> Option<String> {
        self.space.read(cx).selected_model().map(str::to_string)
    }

    /// The model id handed to the most recent submit (read from the space).
    pub fn last_submitted_model(&self, cx: &gpui::App) -> Option<String> {
        self.space
            .read(cx)
            .last_submitted_model()
            .map(str::to_string)
    }

    /// Toggle the model picker (⌥⌘M, or clicking the revealed label).
    /// Resets the keyboard highlight when the picker is opened so arrow
    /// navigation starts fresh. On open, scroll the current model's row into
    /// view so a far-down current selection isn't hidden below the fold of the
    /// capped-height panel.
    pub fn toggle_model_picker(&mut self, cx: &mut Context<Self>) {
        self.model_picker_open = !self.model_picker_open;
        if !self.model_picker_open {
            self.picker_highlighted = None;
        } else if let Some(ix) = self.current_model_index(cx) {
            self.scroll_picker_to(ix);
        }
        cx.notify();
    }

    /// The row index of this space's current model in the picker's model list,
    /// if present. Used to reveal the active selection when the picker opens.
    fn current_model_index(&self, cx: &gpui::App) -> Option<usize> {
        let current = self.current_model(cx);
        self.models
            .read(cx)
            .list()
            .iter()
            .position(|m| m.id == current)
    }

    /// Scroll the keyboard-highlighted picker row into view. The picker rows are
    /// the panel's leading children (one per model, index-aligned), so the model
    /// index doubles as the scroll-child index for `scroll_to_item`.
    fn scroll_highlight_into_view(&mut self) {
        if let Some(ix) = self.picker_highlighted {
            self.scroll_picker_to(ix);
        }
    }

    /// Request the picker's scroll container reveal row `ix`, recording the
    /// target for behavior tests (`ScrollHandle` has no getter for it).
    fn scroll_picker_to(&mut self, ix: usize) {
        self.picker_scroll.scroll_to_item(ix);
        self.last_picker_scroll_target = Some(ix);
    }

    /// Dismiss the model picker without selecting anything (Esc).
    pub fn dismiss_model_picker(&mut self, cx: &mut Context<Self>) {
        self.model_picker_open = false;
        self.picker_highlighted = None;
        cx.notify();
    }

    /// Move the keyboard highlight up one row in the picker, wrapping
    /// from the first row to the last.
    pub fn picker_up(&mut self, cx: &mut Context<Self>) {
        if !self.model_picker_open {
            return;
        }
        let count = self.models.read(cx).list().len();
        if count == 0 {
            return;
        }
        self.picker_highlighted = Some(match self.picker_highlighted {
            // From no selection or already at the top, move to 0 (clamp, not wrap).
            None | Some(0) => 0,
            Some(n) => n - 1,
        });
        self.scroll_highlight_into_view();
        cx.notify();
    }

    /// Move the keyboard highlight down one row in the picker, clamping at
    /// the last row (no wrap).
    pub fn picker_down(&mut self, cx: &mut Context<Self>) {
        if !self.model_picker_open {
            return;
        }
        let count = self.models.read(cx).list().len();
        if count == 0 {
            return;
        }
        self.picker_highlighted = Some(match self.picker_highlighted {
            None => 0,
            Some(n) => (n + 1).min(count - 1),
        });
        self.scroll_highlight_into_view();
        cx.notify();
    }

    /// Confirm the currently highlighted picker row (Enter). No-op when
    /// nothing is highlighted.
    pub fn picker_confirm(&mut self, cx: &mut Context<Self>) {
        if !self.model_picker_open {
            return;
        }
        let Some(idx) = self.picker_highlighted else {
            return;
        };
        let models = self.models.read(cx).list().to_vec();
        if let Some(model) = models.get(idx) {
            let id = model.id.clone();
            self.select_model(id, cx);
        }
    }

    /// The keyboard-highlighted picker row index (if any).
    pub fn picker_highlighted(&self) -> Option<usize> {
        self.picker_highlighted
    }

    /// Test-only: the last row index the picker was asked to scroll into view
    /// (via `scroll_to_item`). Lets behavior tests assert keyboard navigation
    /// keeps the highlighted row visible in the capped-height panel.
    #[doc(hidden)]
    pub fn last_picker_scroll_target_for_test(&self) -> Option<usize> {
        self.last_picker_scroll_target
    }

    /// Choose the model for this space's subsequent sends and close the
    /// picker. A switch while a response is streaming applies to the *next*
    /// send — the in-flight request is never hot-swapped (the selection lives
    /// on the shared `Space`, so both windows see the change).
    pub fn select_model(&mut self, id: String, cx: &mut Context<Self>) {
        self.space.update(cx, |s, cx| s.select_model(id, cx));
        self.model_picker_open = false;
        self.picker_highlighted = None;
        cx.notify();
    }

    /// Persist this window's current model as the config default
    /// (`default_model` override) — the picker's quiet "set as default"
    /// affordance. The picker stays open so the moved "default" marker is
    /// the visible confirmation.
    pub fn set_current_model_as_default(&mut self, cx: &mut Context<Self>) {
        let model = self.current_model(cx);
        self.config
            .update(cx, |c, cx| c.set_default_model(model, cx));
        cx.notify();
    }

    /// Test-only setter for the ⌥-reveal state, so snapshot tests can render
    /// the revealed label without synthesizing platform modifier events.
    /// Writes through to the `WindowInput` entity so `model_revealed` stays
    /// consistent.
    #[doc(hidden)]
    pub fn set_alt_held_for_test(&mut self, alt: bool, cx: &mut Context<Self>) {
        self.window_input
            .update(cx, |wi, cx| wi.set_alt_for_test(alt, cx));
    }

    /// Test-only setter for the picker's open state.
    #[doc(hidden)]
    pub fn set_model_picker_open_for_test(&mut self, open: bool) {
        self.model_picker_open = open;
        if !open {
            self.picker_highlighted = None;
        }
    }

    /// Test-only setter for the keyboard highlight index.
    #[doc(hidden)]
    pub fn set_picker_highlighted_for_test(&mut self, idx: Option<usize>) {
        self.picker_highlighted = idx;
    }
}

impl ChatView {
    /// Construct a chat view. `space_id: None` is the blank page (⌘N): a
    /// fresh space is created lazily by the first exchange. `Some(id)`
    /// reopens an existing space (the Library's path): its messages are
    /// loaded asynchronously on construction and the next exchange
    /// continues that space.
    ///
    /// `window_input` is the per-window modifier entity created by
    /// `open_chat_window`. The root view's `on_modifiers_changed` listener
    /// mirrors every modifier event into it; `ChatView` observes it for the
    /// ⌥ reveal and never registers its own modifier listener.
    pub fn new(
        stores: Stores,
        space_id: Option<String>,
        window_input: Entity<WindowInput>,
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

        let config = stores.config.clone();
        let models = stores.models.clone();
        let account = stores.account.clone();
        let wallet = stores.wallet.clone();
        let spaces = stores.spaces.clone();

        // Get-or-create the shared `Space` entity through the registry.
        // `Some(id)` joins (or creates) the entity for an existing space — and
        // kicks off the one transcript load; a second window opening the same
        // id shares this entity (and its single load), which is the wave-2
        // bug-4 fix. `None` mints a blank space (⌘N) that stays instant and is
        // adopted into the registry once its first exchange assigns an id.
        let space = match space_id {
            Some(id) => spaces.update(cx, |s, cx| s.open(id, cx)),
            None => spaces.update(cx, |s, cx| s.blank(cx)),
        };

        // Onboarding inputs: when an account exists, the stage machine needs
        // a balance snapshot and the active-credential list. The model list
        // is refreshed app-globally at launch (and on `Change::Config`) by
        // `ModelsStore` — the first window reads it from there rather than
        // kicking its own fetch, which is the structural fix for wave-2's
        // starved model list. Each refresh owns its own store task slot, so
        // there is no shared busy gate to dodge.
        let has_account = config
            .read(cx)
            .state()
            .is_some_and(|s| s.has_account && s.has_account_secret);
        if has_account {
            account.update(cx, |s, cx| s.refresh_balances(cx));
            wallet.update(cx, |s, cx| s.refresh(cx));
        }

        // Re-render when the stores the onboarding stage depends on change,
        // and lazily fetch the plans data the first time the derived stage
        // lands on `Plans`. The `Space` gets both a plain `observe` (re-render
        // on any change — transcript reloads, streaming deltas) and a
        // semantic `subscribe` (`SpaceEvent`): tail-scroll keys off
        // `StreamDelta`, and a failed submit (`Failed`) routes the degraded
        // onboarding state, both window-local concerns.
        let _subscriptions = vec![
            cx.observe(&config, |this: &mut Self, _, cx| {
                this.maybe_fetch_plans_data(cx);
                cx.notify();
            }),
            cx.observe(&models, |_, _, cx| cx.notify()),
            cx.observe(&account, |this: &mut Self, _, cx| {
                this.maybe_fetch_plans_data(cx);
                cx.notify();
            }),
            cx.observe(&wallet, |this: &mut Self, _, cx| {
                this.maybe_fetch_plans_data(cx);
                cx.notify();
            }),
            // The `Space` changing (a transcript reload landed, streaming
            // buffers grew, a model selection moved) must re-derive the
            // virtualized item model. `rebuild_transcript(false, …)` reconciles
            // the `list()` item count without forcing a tail re-pin — that is
            // reserved for the semantic `StreamDelta`/submit hooks below.
            cx.observe(&space, |this: &mut Self, _, cx| {
                this.rebuild_transcript(false, cx);
                cx.notify();
            }),
            cx.subscribe_in(&space, window, Self::on_space_event),
            // Re-render when ⌥ transitions so the title-bar reveal responds
            // immediately. The modifier state itself lives in `window_input`;
            // the root view's single listener mirrors it there.
            cx.observe(&window_input, |_, _, cx| cx.notify()),
        ];

        // The transcript renders through gpui's variable-height `list()`,
        // anchored at the *top* (`ListAlignment::Top`) — a book page reads
        // from the top down, and opening a space lands on its first line, the
        // pre-virtualization resting state (the layout pass clamps the scroll
        // to the top when content is shorter than the viewport, and to the
        // first item otherwise). Overdraw of ~one viewport keeps scrolling
        // smooth without measuring the whole transcript.
        //
        // Tail-follow (`FollowMode::Tail`) is engaged on *submit* (see
        // `submit`), not at construction: a reopened space must rest at the
        // top, and `set_follow_mode(Tail)` would immediately jump the scroll
        // to the end. Once engaged, the list re-pins to the bottom on growth
        // while the reader is at the bottom and disengages the instant they
        // scroll up — exactly the tail policy.
        let list_state = ListState::new(0, ListAlignment::Top, px(800.));

        let mut this = Self {
            stores,
            config,
            models,
            account,
            wallet,
            space,
            prompt_editor,
            error: None,
            onboarding: OnboardingFlow::default(),
            show_plans_after_error: false,
            window_input,
            model_picker_open: false,
            list_state,
            transcript_items: Vec::new(),
            participant_indicator: None,
            last_reconcile_was_reset: None,
            picker_highlighted: None,
            picker_scroll: ScrollHandle::new(),
            last_picker_scroll_target: None,
            focus_handle,
            _subscriptions,
        };
        // Seed the list's item model from whatever the (possibly preloaded)
        // space already holds, so the first frame renders the right rows.
        this.rebuild_transcript(false, cx);
        this
    }

    /// React to a semantic `SpaceEvent` from the shared `Space`. Plain
    /// re-render and item-model reconciliation are handled by the sibling
    /// `observe`; here we handle the window-local *tail policy* and the
    /// degraded-onboarding routing on a typed failure.
    ///
    /// Tail policy (the book-page "writing scrolls under your hand" feel):
    /// the `list()` runs in `FollowMode::Tail`, so it auto-snaps to the
    /// bottom while the reader is at the bottom and disengages the instant
    /// they scroll up. On a streaming delta the streaming item's height grew,
    /// so we `remeasure` it and, *if still following the tail*, `scroll_to_end`
    /// so the freshest token stays visible — a reader who has scrolled up is
    /// left undisturbed (`is_following_tail()` is false for them).
    fn on_space_event(
        &mut self,
        _space: &Entity<Space>,
        event: &SpaceEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            SpaceEvent::StreamDelta => {
                // The streaming row's content grew in place; its item index is
                // stable (same `Streaming` slot), so remeasure rather than
                // splice. Re-pin to the tail only while actively following.
                if let Some(ix) = self.streaming_item_index() {
                    self.list_state.remeasure_items(ix..ix + 1);
                }
                if self.list_state.is_following_tail() {
                    self.list_state.scroll_to_end();
                }
                cx.notify();
            }
            SpaceEvent::MessagesChanged | SpaceEvent::StreamEnded => {
                // The transcript shape changed (a turn appended/finalized).
                // Reconcile the item model; a still-following reader re-pins.
                self.rebuild_transcript(true, cx);
                cx.notify();
            }
            SpaceEvent::Failed(e) => {
                self.apply_chat_failure(e.clone(), cx);
            }
        }
    }

    /// The list-item index of the in-flight streaming row, if present.
    fn streaming_item_index(&self) -> Option<usize> {
        self.transcript_items
            .iter()
            .position(|item| matches!(item, TranscriptItem::Streaming))
    }

    /// The delim voice (label + error-ness) that introduces the transcript
    /// item at `ix` — i.e. whose turn that row belongs to. `None` for the
    /// composer's own "You" frame and for an out-of-range index (the composer
    /// is its own writing zone, not someone else's turn to surface).
    fn item_voice(&self, ix: usize, cx: &gpui::App) -> Option<ParticipantIndicator> {
        match self.transcript_items.get(ix)? {
            TranscriptItem::Message { index, .. } => {
                let role = self
                    .space
                    .read(cx)
                    .messages()
                    .get(*index)?
                    .message
                    .role
                    .clone();
                Some(ParticipantIndicator {
                    label: participant_label(&role),
                    is_error: role == "error",
                })
            }
            TranscriptItem::Streaming => Some(ParticipantIndicator {
                label: participant_label("assistant"),
                is_error: false,
            }),
            TranscriptItem::Error => Some(ParticipantIndicator {
                label: participant_label("error"),
                is_error: true,
            }),
            TranscriptItem::Composer { .. } => None,
        }
    }

    /// Derive the persistent participant indicator from the `list()` scroll
    /// state and the item model — the cue that tells the reader *whose turn*
    /// they're reading once the chapter delim has scrolled off the top.
    ///
    /// It is visible exactly when the top of the viewport sits **inside** a
    /// message whose leading delim has scrolled off, and hidden the moment
    /// the real delim (or the page top) is visible — so it fills in only when
    /// the page-local cue is missing, never competing with it. Concretely,
    /// for the item at the logical scroll top:
    ///
    /// - at the very top of the page (`item_ix == 0`, `offset == 0`) → hidden;
    /// - if the scroll offset into that item is within the leading-delim band
    ///   (the delim is on screen, or we're at the item's top edge) → hidden;
    /// - otherwise (scrolled past the delim, or into a delim-less first
    ///   message) → show that item's voice.
    ///
    /// Returns `None` over the composer (the writing zone is the reader's own
    /// frame, not a turn to label). Pure function of the arguments, exercised
    /// directly by [`Self::derive_participant_indicator_at`] in tests.
    fn derive_participant_indicator(&self, cx: &gpui::App) -> Option<ParticipantIndicator> {
        // No measured layout yet (item_count 0, or never painted): no cue.
        if self.list_state.item_count() == 0 {
            return None;
        }
        let top = self.list_state.logical_scroll_top();
        self.derive_participant_indicator_at(top.item_ix, top.offset_in_item, cx)
    }

    /// The pure core of [`Self::derive_participant_indicator`], split out so a
    /// behavior test can drive the visibility rule directly from a synthetic
    /// scroll position without a real layout pass.
    fn derive_participant_indicator_at(
        &self,
        item_ix: usize,
        offset_in_item: gpui::Pixels,
        cx: &gpui::App,
    ) -> Option<ParticipantIndicator> {
        // At the page top, the delim (or the page's first line) is visible.
        if item_ix == 0 && offset_in_item <= px(0.) {
            return None;
        }
        // The leading-delim band: a delim row is `pt_8 + label + pb_6` tall.
        // While the scroll offset into the top item is still within roughly
        // that band, the delim itself is on (or just leaving) the screen, so
        // the page-local cue is present and the indicator stays hidden. Once
        // we've scrolled clearly past it, the cue is gone and we surface it.
        // A delim-less first message (`item_ix == 0`) has no band — any
        // offset past the top means its opening line has scrolled up.
        let item = self.transcript_items.get(item_ix)?;
        let has_leading_delim = match item {
            TranscriptItem::Message { leading_delim, .. } => *leading_delim,
            TranscriptItem::Composer { leading_delim } => *leading_delim,
            TranscriptItem::Streaming | TranscriptItem::Error => true,
        };
        if has_leading_delim && offset_in_item < DELIM_BAND_HEIGHT {
            return None;
        }
        self.item_voice(item_ix, cx)
    }

    /// Read-only access to the derived indicator, for behavior tests.
    pub fn participant_indicator(&self) -> Option<&ParticipantIndicator> {
        self.participant_indicator.as_ref()
    }

    /// Test-only: drive the visibility rule from a synthetic scroll position
    /// without a real layout pass.
    #[doc(hidden)]
    pub fn participant_indicator_at_for_test(
        &self,
        item_ix: usize,
        offset_px: f32,
        cx: &gpui::App,
    ) -> Option<ParticipantIndicator> {
        self.derive_participant_indicator_at(item_ix, px(offset_px), cx)
    }

    /// Test-only: the current flat transcript item count (composer included).
    #[doc(hidden)]
    pub fn transcript_item_count_for_test(&self) -> usize {
        self.transcript_items.len()
    }

    /// Test-only: the `ListState`'s own item count. The regression repro
    /// asserts this stays in lockstep with [`Self::transcript_item_count_for_test`]
    /// — a desync is exactly the disappearing-message bug (the model has the
    /// item but the list never measures/renders it).
    #[doc(hidden)]
    pub fn list_state_item_count_for_test(&self) -> usize {
        self.list_state.item_count()
    }

    /// Test-only: how the last `rebuild_transcript` reconciled the list.
    /// `Some(true)` = wholesale `reset`, `Some(false)` = tail-append fast-path,
    /// `None` = early-returned (model unchanged). The regression replay asserts
    /// a suffix-shifting reshape took the `reset` path, not a bogus tail-splice.
    #[doc(hidden)]
    pub fn last_reconcile_was_reset_for_test(&self) -> Option<bool> {
        self.last_reconcile_was_reset
    }

    /// Test-only: the message indices currently present in the flat transcript
    /// model, in order. Used by the regression repro to assert the user's first
    /// message is reflected as a rendered `Message` item (not just persisted).
    #[doc(hidden)]
    pub fn transcript_message_indices_for_test(&self) -> Vec<usize> {
        self.transcript_items
            .iter()
            .filter_map(|item| match item {
                TranscriptItem::Message { index, .. } => Some(*index),
                _ => None,
            })
            .collect()
    }

    /// Test-only: scroll the transcript list so item `item_ix` is at the top
    /// of the viewport, `offset_px` pixels into it. Used by snapshot cases to
    /// render the scrolled state that surfaces the participant indicator.
    /// (The list must have been painted at least once for the offset to take
    /// full effect; snapshot cases render after `run_until_parked`.)
    #[doc(hidden)]
    pub fn scroll_transcript_for_test(
        &mut self,
        item_ix: usize,
        offset_px: f32,
        cx: &mut Context<Self>,
    ) {
        self.list_state.scroll_to(gpui::ListOffset {
            item_ix,
            offset_in_item: px(offset_px),
        });
        cx.notify();
    }

    /// Test-only: render exactly the items in `range` (what the `list()`
    /// closure does for the visible window) and return how many produced a
    /// non-empty element — the O(visible) per-frame work, for the perf guard.
    #[doc(hidden)]
    pub fn render_transcript_window_for_test(
        &mut self,
        range: std::ops::Range<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> usize {
        let mut n = 0;
        for ix in range {
            if ix < self.transcript_items.len() {
                let _ = self.render_transcript_item(ix, window, cx);
                n += 1;
            }
        }
        n
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
    pub fn onboarding_stage(&self, cx: &gpui::App, composer_empty: bool) -> OnboardingStage {
        let space = self.space.read(cx);
        if !space.messages().is_empty()
            || space.is_streaming()
            || self.error.is_some()
            || !composer_empty
        {
            return OnboardingStage::Ready;
        }
        let config = self.config.read(cx);
        let Some(state) = config.state() else {
            return OnboardingStage::Ready;
        };
        if !state.has_account || !state.has_account_secret {
            return OnboardingStage::Welcome;
        }
        if self.onboarding.dismissed {
            return OnboardingStage::Ready;
        }
        let balances = self.account.read(cx).balances();
        let credentials_empty = self.wallet.read(cx).credentials().is_empty();
        let balance_known_zero = balances.value().is_some_and(|b| b.available <= 0);
        if balance_known_zero && credentials_empty {
            return OnboardingStage::Plans;
        }
        if self.onboarding.entered_plans && balances.value().is_none() {
            return OnboardingStage::Plans;
        }
        OnboardingStage::Ready
    }

    fn current_stage(&self, cx: &Context<Self>) -> OnboardingStage {
        let composer_empty = self.prompt_editor.read(cx).state.markdown.trim().is_empty();
        self.onboarding_stage(cx, composer_empty)
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
        if self.account.read(cx).prices().has_value() {
            self.onboarding.plans_fetch_attempted = true;
            return;
        }
        // Latch unconditionally now: each store refresh owns its own task
        // slot (no shared busy gate), so a `Plans`-stage prices fetch always
        // starts. We still guard against re-entry via the latch.
        self.account.update(cx, |s, cx| {
            s.refresh_prices(cx);
            s.refresh_balances(cx);
        });
        self.onboarding.plans_fetch_attempted = true;
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

        // Awaitable account-create: the onboarding flow owns the await in its
        // own (detached, view-scoped) task and refreshes config/account on
        // success. `None` on stub stores (behavior tests): the in-flight flag
        // above is the observable state-machine transition.
        let Some(rx) = self.account.read(cx).request_account_create() else {
            return;
        };
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
                        // Move straight to the plans step: refresh the config
                        // snapshot (has_account flips true) and fetch prices +
                        // the initial (zero) balance. Each refresh owns its own
                        // store task slot, so the calls always start.
                        this.onboarding.entered_plans = true;
                        this.config.update(cx, |c, cx| c.refresh(cx));
                        this.account.update(cx, |s, cx| {
                            s.refresh_prices(cx);
                            s.refresh_balances(cx);
                        });
                        this.onboarding.plans_fetch_attempted = true;
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

        let Some(rx) = self.account.read(cx).request_checkout(price_id) else {
            return;
        };
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
        // Stub stores have no backend to poll.
        if self.stores.app_core().is_none() {
            return;
        }
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
                // Ask the AccountStore for a fresh balances fetch via its
                // awaitable `request_balances` — the poll (the initiator)
                // owns the await here and writes the result back into the
                // store, since the poll itself is the source of this change.
                let Some(rx) = this
                    .read_with(cx, |this, cx| this.account.read(cx).request_balances())
                    .ok()
                    .flatten()
                else {
                    break;
                };
                let res = match rx.await {
                    Ok(res) => res,
                    Err(_) => break, // task cancelled
                };
                let stop = this
                    .update(cx, |this, cx| {
                        match res {
                            Ok(balances) => {
                                this.onboarding.plans_error = None;
                                if balances.available > 0 {
                                    this.onboarding.awaiting_checkout = false;
                                }
                                this.account
                                    .update(cx, |s, cx| s.set_balances(balances, cx));
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

    /// Route a failed chat submit. Called from the space's `SpaceEvent::Failed`
    /// handler. `InsufficientBalance` additionally surfaces the plans list
    /// below the transcript via the error band — typed routing, not string
    /// matching. The streaming state itself is cleared by the `Space` (it owns
    /// it); this method only sets the *window-local* error band + degraded
    /// onboarding presentation.
    pub fn apply_chat_failure(&mut self, e: AppError, cx: &mut Context<Self>) {
        // Look through app-core's `ChatFailed` id-carrying wrapper before
        // routing on variant. The `Space` already emits the unwrapped source,
        // but `root()` keeps this correct if a wrapped error ever reaches here
        // directly.
        let root = e.root();
        if matches!(root, AppError::InsufficientBalance { .. }) {
            self.show_plans_after_error = true;
            if !self.account.read(cx).prices().has_value() {
                self.account.update(cx, |s, cx| {
                    s.refresh_prices(cx);
                    s.refresh_balances(cx);
                });
            }
        }
        self.error = Some(root.to_string());
        // The error band is a window-local transcript item; reconcile the
        // model so it appears, and bring it into view (it's the new tail).
        self.rebuild_transcript(true, cx);
        cx.notify();
    }

    fn submit(&mut self, _: &Send, _window: &mut Window, cx: &mut Context<Self>) {
        // Submit-during-streaming is a no-op (the current UX); the `Space`'s
        // runner slot enforces this structurally, so we read it before
        // touching the composer.
        if self.space.read(cx).is_streaming() {
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

        // Resolve the model at submit time (space selection → config default →
        // embedded fallback). The `Space` records it before its own backend
        // guard, so stub-core tests observe exactly what a real send would use.
        let model = self.current_model(cx);

        self.prompt_editor.update(cx, |editor, cx| {
            editor.state = EditorState::default();
            cx.notify();
        });
        // Clear the window-local error band on a fresh submit. The transcript
        // append + streaming entry + the streaming runner all live on the
        // shared `Space`, so a second window on the same space sees them too.
        self.error = None;
        self.show_plans_after_error = false;
        self.space.update(cx, |s, cx| {
            s.submit(prompt, model, cx);
        });
        // Submit always brings the new exchange into view, regardless of where
        // the reader had scrolled — re-engage tail follow and pin to the
        // bottom. `Space::submit` appended the user turn + streaming row
        // synchronously, so its `MessagesChanged` already reconciled the item
        // model; here we assert the scroll intent. (Re-arming `Tail` also
        // re-engages following if a prior scroll-up had disengaged it.)
        self.list_state.set_follow_mode(gpui::FollowMode::Tail);
        self.list_state.scroll_to_end();
        cx.notify();
    }

    /// Recompute the flat `TranscriptItem` model from the shared `Space` plus
    /// the window-local error/plans state, and reconcile the `list()`'s item
    /// count against it. This is the chat analogue of the Record/Library flat
    /// display model: the `list()` render closure is a dumb indexer over
    /// `transcript_items`, so per-frame work stays O(visible).
    ///
    /// Reconciliation is **correctness-first**. The old append-aware heuristic
    /// compared a *prefix* of the new model against the old one — but in this
    /// UI the dynamic suffix (streaming row, error band, and the always-trailing
    /// composer) moves on *every* real transition, so that prefix test was
    /// simultaneously (a) a no-op fast-path that never actually fired for a turn
    /// append, and (b) a latent footgun: any reshape that *looked* like a tail
    /// append (`[Composer] → [Message, Streaming, Composer]` on the first
    /// submit — the composer is the last item, so new items appear *before* it,
    /// not after) would `splice` only a bogus tail and leave `ListState`'s item
    /// count and per-item measurements out of step with `transcript_items`. A
    /// desynced list never measures or paints the items it lost track of —
    /// which is exactly the regression where the first user message vanishes
    /// from the transcript the instant streaming begins (it is persisted fine;
    /// the list just stopped rendering it). Regression replay:
    /// `first_message_survives_streaming_start` in `tests/behavior.rs`.
    ///
    /// The fix: the fast-path fires **only** when the old model is a *complete*
    /// prefix of the new one — i.e. the entire previous item sequence, suffix
    /// included, is unchanged and items were appended strictly *after* it. That
    /// is provably a tail-append, so splicing only the new tail preserves the
    /// reader's exact scroll position. Every other reshape (any change anywhere
    /// before the new tail — a turn inserted ahead of the composer, the
    /// streaming row appearing/finalizing, an error band toggling, the
    /// first-message delim flipping, a reload reshaping the middle) takes the
    /// `reset` path, which rebuilds the list's item set wholesale so the count
    /// and measurements can never drift from the model.
    ///
    /// `pin_tail` (set by submit / failure / stream-end, and propagated from
    /// `is_following_tail` on the model-only `observe` path) re-pins to the
    /// bottom after reconciliation so the freshest content stays visible; a
    /// reader scrolled up mid-history is never yanked.
    ///
    /// The **composer is the final item** and is passed to `splice_focusable`
    /// with the editor's focus handle, so when it scrolls offscreen while
    /// focused the `list()` keeps rendering it for keyboard interaction — the
    /// single-scroll-surface, book-page feel under virtualization.
    fn rebuild_transcript(&mut self, pin_tail: bool, cx: &mut Context<Self>) {
        let new_items = self.compute_transcript_items(cx);
        let in_sync = self.list_state.item_count() == self.transcript_items.len();
        if new_items == self.transcript_items && in_sync {
            // Item model unchanged (e.g. a re-render-only notify): nothing to
            // reconcile, and we leave `last_reconcile_was_reset` untouched so a
            // no-op notify firing right after a real reshape doesn't erase the
            // record of how that reshape reconciled. A still-following stream
            // re-pin is handled by the StreamDelta hook, not here.
            return;
        }

        let was_following = self.list_state.is_following_tail();
        let composer_focus = self.prompt_editor.read(cx).focus_handle(cx);

        // A provable tail-append: the *entire* old model is an unchanged prefix
        // of the new one (suffix included), so only items strictly after it were
        // added. This is the one case where we can splice just the new tail and
        // keep the reader's scroll position pixel-exact. Crucially it requires
        // the list to already be in sync with the old model — otherwise we fall
        // through to `reset`, which re-establishes the invariant from scratch.
        let old_len = self.transcript_items.len();
        let is_tail_append = in_sync
            && new_items.len() > old_len
            && new_items[..old_len] == self.transcript_items[..];

        self.last_reconcile_was_reset = Some(!is_tail_append);
        if is_tail_append {
            let focus_handles = (old_len..new_items.len()).map(|ix| {
                matches!(new_items[ix], TranscriptItem::Composer { .. })
                    .then(|| composer_focus.clone())
            });
            self.list_state
                .splice_focusable(old_len..old_len, focus_handles);
        } else {
            // Any reshape (or a list/model desync): rebuild the list's item set
            // wholesale so its count and measurements match the new model
            // exactly. `reset` already replaces every item with an unmeasured
            // copy, so we splice the focus handles over that fresh set rather
            // than resetting and re-splicing the whole range twice.
            let count = new_items.len();
            self.list_state.reset(count);
            let focus_handles = (0..count).map(|ix| {
                matches!(new_items[ix], TranscriptItem::Composer { .. })
                    .then(|| composer_focus.clone())
            });
            self.list_state.splice_focusable(0..count, focus_handles);
        }

        self.transcript_items = new_items;

        if pin_tail {
            self.list_state.set_follow_mode(gpui::FollowMode::Tail);
            self.list_state.scroll_to_end();
        } else if was_following {
            // Preserve an at-bottom reader's tail follow across a reshape.
            self.list_state.scroll_to_end();
        }
    }

    /// Build the flat transcript model (no `ListState` mutation — pure).
    /// Mirrors the order `render_transcript_item` renders: messages, then the
    /// streaming row, then the error band, then the composer.
    fn compute_transcript_items(&self, cx: &Context<Self>) -> Vec<TranscriptItem> {
        let space = self.space.read(cx);
        let message_count = space.messages().len();
        let streaming = space.is_streaming();
        let has_error = self.error.is_some();

        let mut items = Vec::with_capacity(message_count + 2);
        for index in 0..message_count {
            // The very first row of the page has no leading delim (the user's
            // opening text is the start of the page); every later turn does.
            items.push(TranscriptItem::Message {
                index,
                leading_delim: index > 0,
            });
        }
        if streaming {
            items.push(TranscriptItem::Streaming);
        }
        if has_error {
            items.push(TranscriptItem::Error);
        }
        // The composer is always the final item. It gets a leading "You" delim
        // only when there is preceding content (otherwise the cursor sits at
        // the top of a blank page).
        let has_preceding = message_count > 0 || streaming || has_error;
        items.push(TranscriptItem::Composer {
            leading_delim: has_preceding,
        });
        items
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

        // Re-derive the persistent participant indicator from the `list()`
        // scroll state each frame (scroll re-renders flow through here). Only
        // meaningful on the transcript stage; the onboarding pages have no
        // transcript to read a voice from. Storing it keeps the title-bar
        // render reading one consistent value and lets behavior tests assert
        // it. No layout shift — the band is absolute chrome.
        self.participant_indicator = if stage == OnboardingStage::Ready {
            self.derive_participant_indicator(cx)
        } else {
            None
        };

        let content: gpui::AnyElement = match stage {
            OnboardingStage::Welcome => self.render_welcome(window, cx).into_any_element(),
            OnboardingStage::Plans => self.render_plans_page(window, cx).into_any_element(),
            OnboardingStage::Ready => self.render_transcript(window, cx).into_any_element(),
        };

        let theme = cx.theme();
        let wi = self.window_input.clone();
        v_flex()
            .key_context("ChatView")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::submit))
            .on_action(
                cx.listener(|this, _: &ToggleModelPicker, _, cx| this.toggle_model_picker(cx)),
            )
            // Picker keyboard navigation — no-ops when the picker is closed.
            .on_action(
                cx.listener(|this, _: &DismissModelPicker, _, cx| this.dismiss_model_picker(cx)),
            )
            .on_action(cx.listener(|this, _: &PickerUp, _, cx| this.picker_up(cx)))
            .on_action(cx.listener(|this, _: &PickerDown, _, cx| this.picker_down(cx)))
            .on_action(cx.listener(|this, _: &PickerConfirm, _, cx| this.picker_confirm(cx)))
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            // Single modifier listener for this window: mirrors every event
            // into `WindowInput` so all descendant views observe it rather
            // than registering their own listeners on non-ancestor elements.
            .on_modifiers_changed(cx.listener(move |_, event: &ModifiersChangedEvent, _, cx| {
                wi.update(cx, |wi, cx| {
                    wi.update_modifiers(event, cx);
                });
            }))
            .relative()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(content)
            .child(self.render_title_bar(stage, cx))
    }
}

impl ChatView {
    /// The normal page: transcript + composer in one virtualized scroll
    /// surface. The transcript renders through gpui's variable-height
    /// `list()` (`docs/architecture/state.md` → "Lists"): the element holds
    /// the persistent `ListState` scroll handle and the render closure is a
    /// dumb indexer over the precomputed `transcript_items` flat model, so
    /// per-frame work is O(visible), not O(messages). The composer is the
    /// final list item, preserving the single-scroll-surface, book-page feel.
    ///
    /// The leading edge under the traffic lights is the list's top *padding*
    /// (accounted in the list's own layout), replacing the old
    /// `pt(TITLE_BAR_RESERVE)` on a column inside an `overflow_y_scroll` div.
    /// `.size_full()` makes the list fill its flex parent and scroll
    /// internally (`ListSizingBehavior::Auto`); it must **not** be wrapped in
    /// another scroll container.
    fn render_transcript(&self, _window: &Window, cx: &Context<Self>) -> gpui::Stateful<Div> {
        let weak = cx.entity().downgrade();
        let list_state = self.list_state.clone();
        let item_count = self.transcript_items.len();

        let list = list(list_state, move |ix, window, cx| {
            weak.upgrade()
                .map(|view| view.update(cx, |this, cx| this.render_transcript_item(ix, window, cx)))
                .unwrap_or_else(|| div().into_any_element())
        })
        .with_sizing_behavior(gpui::ListSizingBehavior::Auto)
        .pt(TITLE_BAR_RESERVE)
        .size_full();

        // Wrap in a stable-id stateful div so the return type matches the
        // other `Render` arms and so the list participates in the page's
        // `relative` stacking with the absolute title-bar band painted over
        // it. The div carries no scroll — the `list()` owns scrolling.
        div()
            .id("transcript")
            .w_full()
            .flex_1()
            .min_h_0()
            .child(if item_count == 0 {
                div().into_any_element()
            } else {
                list.into_any_element()
            })
    }

    /// Render one virtualized transcript item. Dispatches on the flat model;
    /// the chapter delim that introduces a turn is rendered *inside* the same
    /// item as its body so the two measure and scroll as one unit (and a
    /// delim never lands alone on an item where it could collapse).
    fn render_transcript_item(
        &mut self,
        ix: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(item) = self.transcript_items.get(ix).cloned() else {
            return div().into_any_element();
        };
        match item {
            TranscriptItem::Message {
                index,
                leading_delim,
            } => self.render_message_item(index, leading_delim, cx),
            TranscriptItem::Streaming => self.render_streaming_item(cx),
            TranscriptItem::Error => self.render_error_item(cx),
            TranscriptItem::Composer { leading_delim } => {
                self.render_composer_item(leading_delim, window, cx)
            }
        }
    }

    /// A persisted transcript message (with its leading chapter delim unless
    /// it is the first row of the page). One `SpaceMessage` = exactly one
    /// item, regardless of how many markdown blocks its content parses into.
    /// The `("msg", idx)` / `("thinking-body", idx)` TextView ids are stable
    /// per message index, so the markdown parse is a one-time cost that
    /// survives list-item reuse (TextView early-returns on unchanged text).
    fn render_message_item(
        &self,
        idx: usize,
        leading_delim: bool,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let markdown_style = markdown_style(theme.mode.is_dark());
        let Some(entry) = self.space.read(cx).messages().get(idx).cloned() else {
            return div().into_any_element();
        };
        let msg = &entry.message;

        let mut container = v_flex().w_full().gap_0();

        // Chapter delimiter — a hairline rule + italic participant name.
        // The very first row has no leading delim (the user's opening text is
        // the start of the page); every later turn does. Errors use
        // `theme.danger` for the label so the chrome itself signals the role.
        if leading_delim {
            let label_color = if msg.role == "error" {
                theme.danger
            } else {
                theme.muted_foreground
            };
            container = container.child(chapter_delim(
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
            // Wrap the Button in an `h_flex` so it sizes to its content;
            // without a flex-row parent the button stretches to v_flex's full
            // cross-axis width and the label ends up center-aligned. Centered
            // inside a `prose_row` so the disclosure aligns with the reading
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
                                    this.space.update(cx, |s, cx| {
                                        s.toggle_message_reasoning(idx, cx);
                                    });
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

        container.child(row).into_any_element()
    }

    /// The in-flight streaming assistant row (delim + thinking disclosure +
    /// partial body). The `("streaming-body", 0)` / `("streaming-thinking-
    /// body", 0)` TextView ids are stable across deltas, so the partial
    /// markdown re-parses only because its text grew — never spuriously.
    fn render_streaming_item(&self, cx: &Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let markdown_style = markdown_style(theme.mode.is_dark());
        let Some(s) = self.space.read(cx).streaming().cloned() else {
            return div().into_any_element();
        };
        let fg = theme.foreground;
        let muted_fg = theme.muted_foreground;
        let danger = theme.danger;

        let mut container = v_flex().w_full().gap_0();

        // Same chapter delimiter pattern as a finalized assistant message; the
        // row underneath fills in as deltas arrive.
        container = container.child(chapter_delim(
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

        // Disclosure only appears once reasoning has actually arrived; before
        // that we just show a "Thinking…" status so the user sees something
        // is in flight.
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
                                    this.space.update(cx, |s, cx| {
                                        s.toggle_streaming_reasoning(cx);
                                    });
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
            // No reasoning *and* no content yet — show the "still working"
            // status indicator. Plain Label, no toggle, no markdown plumbing.
            // Aligned with the prose column so the status doesn't visually
            // jump when content arrives.
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

        container.child(col).into_any_element()
    }

    /// The window-local error band (delim + body + optional below-band plans
    /// list when a submit failed with `InsufficientBalance`).
    fn render_error_item(&self, cx: &Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let markdown_style = markdown_style(theme.mode.is_dark());
        let Some(err) = self.error.clone() else {
            return div().into_any_element();
        };

        let mut container = v_flex().w_full().gap_0();

        // Errors render through `TextView::markdown` (not a raw `SharedString`)
        // so the user can select and copy the text. The chapter delim's
        // "Error" label is in `theme.danger`, and the body inherits the danger
        // color through the row's `text_color`.
        container = container.child(chapter_delim(
            participant_label("error"),
            theme.border,
            theme.danger,
        ));
        container = container.child(
            div().w_full().px_5().text_color(theme.danger).child(
                prose_row().child(
                    prose().child(
                        TextView::markdown(("chat-error", 0usize), err)
                            .selectable(true)
                            .style(markdown_style.clone()),
                    ),
                ),
            ),
        );

        // A submit that failed with `InsufficientBalance` surfaces the plans
        // right here, below the transcript's error band — the same hairline
        // list the onboarding plans page uses, not a modal.
        if self.show_plans_after_error {
            container = container.child(
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

        container.into_any_element()
    }

    /// The composer item — the final list item. A "You" chapter delim sits
    /// above it whenever there's preceding content (mirroring the way earlier
    /// turns are introduced); on a fresh, empty page the delim is omitted so
    /// the cursor sits cleanly at the top.
    ///
    /// The editor renders one gpui block per markdown block, so it grows
    /// naturally with content; the enclosing `list()` handles overflow as one
    /// continuous unit. `min_h(...)` keeps the empty editor clickable even
    /// when its markdown is "" (the render pipeline emits no blocks then, so
    /// without a floor the editor collapses to zero height and the user can't
    /// click back into it after losing focus); the floor is one body line at
    /// the prose line-height ratio.
    ///
    /// **Half-viewport bottom padding** below the editor keeps the cursor off
    /// the bottom edge: there's always a half-page of empty space below the
    /// active line, so the list's tail-follow keeps the typing zone in the
    /// comfortable middle of the viewport as content grows. Computed from the
    /// live viewport height so it tracks window resizes.
    fn render_composer_item(
        &self,
        leading_delim: bool,
        window: &mut Window,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let composer_pb = window.viewport_size().height * 0.5;
        let composer_min_h = PROSE_FONT_SIZE * PROSE_LINE_HEIGHT;

        let mut container = v_flex().w_full().gap_0();
        if leading_delim {
            container = container.child(chapter_delim(
                participant_label("user"),
                theme.border,
                theme.muted_foreground,
            ));
        }
        container
            .child(
                div().w_full().px_5().pb(composer_pb).child(
                    prose_row().child(
                        prose()
                            .min_h(composer_min_h)
                            .child(self.prompt_editor.clone()),
                    ),
                ),
            )
            .into_any_element()
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
        let account = self.account.read(cx);
        let prices_cell = account.prices();
        let prices = prices_cell.value().cloned().unwrap_or_default();
        let prices_loading = prices_cell.is_loading() || prices_cell.is_stale();
        let prices_error = prices_cell.error().map(|e| e.to_string());
        let ob = &self.onboarding;

        let mut list = v_flex().w_full();

        if prices.is_empty() {
            if prices_loading {
                // A prices refresh is in flight.
                list = list.child(
                    div()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .child("Loading plans…"),
                );
            } else if let Some(err) = prices_error {
                list = list
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.danger)
                            .child(SharedString::from(err)),
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
                                this.account.update(cx, |s, cx| {
                                    s.refresh_prices(cx);
                                    s.refresh_balances(cx);
                                });
                                this.onboarding.plans_fetch_attempted = true;
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

        // The rows themselves are shared with the Settings Account pane —
        // see `crate::plans::plan_rows`. The click handler routes back into
        // this view through a weak entity handle (`plan_rows` is a free
        // function, so it can't take `cx.listener` directly).
        let weak = cx.entity().downgrade();
        let on_select: crate::plans::PlanSelectHandler =
            std::rc::Rc::new(move |price_id, _window, app| {
                let _ = weak.update(app, |this, cx| this.begin_checkout(price_id, cx));
            });
        list = list.child(crate::plans::plan_rows(
            &prices,
            ob.checkout_pending.as_deref(),
            on_select,
            cx,
        ));

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

/// Approximate rendered height of a `chapter_delim` row — its `pt_8` (32px) +
/// the `text_sm` label line + `pb_6` (24px). The persistent participant
/// indicator stays hidden while the scroll offset into the viewport-top item
/// is still within this band (the real delim is on or just leaving the
/// screen), and surfaces once we've scrolled clearly past it. Slightly
/// conservative so the hand-off from delim to indicator has no visible gap
/// where neither cue is present.
const DELIM_BAND_HEIGHT: gpui::Pixels = gpui::px(72.);

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
        // Inline code shares its shaped line with Newsreader body text, and
        // gpui can't size a single run independently (`TextRun` has no font
        // size), so the ~0.9× inline-code size good typography wants is
        // approximated by *family* instead: Courier New's x-height (0.423 em)
        // matches Newsreader's (0.426 em) almost exactly, where the theme's
        // Menlo (0.547 em) reads ~28% larger than the surrounding words.
        // Fenced code blocks keep Menlo — they're whole lines of mono with no
        // serif neighbors, so density and clarity win there.
        .inline_code_font_family("Courier New")
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

impl ChatView {
    /// Title-bar band: a gradient that fades from full `theme.background` at
    /// the top to fully transparent at the bottom of the reserve. Painted
    /// over the scroll area (positioned absolutely, last child wins z-order
    /// in gpui), so messages scrolling up under it dissolve smoothly instead
    /// of clipping.
    ///
    /// Two non-aesthetic modifiers tame the title-bar band:
    ///
    /// - `.cursor_default()` sets the platform cursor to `Arrow` over the
    ///   band *and* causes gpui to register a hitbox here
    ///   (`Interactivity::should_insert_hitbox` includes
    ///   `style.mouse_cursor.is_some()`). Without it, the text below keeps
    ///   winning the cursor-style lookup and the I-beam shows over the band.
    /// - `.block_mouse_except_scroll()` upgrades that hitbox to swallow
    ///   click and drag events so a double-click-drag in the band doesn't
    ///   fall through to `TextView`'s selectable handler and start a text
    ///   selection underneath. Scroll passes through, so wheel-scrolling
    ///   while the cursor is in the band still scrolls the chat.
    ///
    /// macOS native titlebar behavior (drag, double-click-to-zoom) is
    /// handled by AppKit at the NSWindow layer before the gpui content view
    /// is asked, so blocking mouse on the gpui side doesn't disturb it.
    ///
    /// The band hosts two quiet, coexisting cues, one per side:
    ///
    /// - **Right** — the ⌥-revealed model label (and the picker it anchors).
    ///   At rest the band is pure gradient (the page stays sacred); while ⌥ is
    ///   held — or the picker is open — the current model id appears
    ///   right-aligned in `text_sm` muted italic. The right side is the
    ///   *power-user, on-demand* cue.
    /// - **Left** — the **persistent participant indicator**: when the top of
    ///   the viewport has scrolled into a turn whose chapter delim is off
    ///   screen, the delim's voice ("You" / "Eidola" / "Error") fades in
    ///   left-aligned in that same `text_sm` muted-italic voice, so a reader
    ///   deep in a long answer still knows whose turn they're reading. It is
    ///   hidden at the page top and whenever a real delim is on screen (see
    ///   [`Self::derive_participant_indicator`]); error turns keep the danger
    ///   color. The chapter delim identifies the speaker *at the boundary*;
    ///   this indicator carries that identity *forward* while the boundary is
    ///   off screen — the two are designed as one system (the delim owns the
    ///   in-page cue, the indicator the out-of-page one) so they never compete.
    ///
    /// Both cues are absolutely positioned chrome painted over the scroll
    /// area, so neither can shift the page layout (no reflow on reveal/scroll).
    /// They render only on the `Ready` stage — the onboarding pages have
    /// neither a transcript voice nor sends to describe.
    fn render_title_bar(&self, stage: OnboardingStage, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let bg = theme.background;
        let mut band = h_flex()
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(TITLE_BAR_RESERVE)
            .justify_end()
            .items_center()
            .cursor_default()
            .block_mouse_except_scroll()
            .bg(linear_gradient(
                180.,
                linear_color_stop(bg, 0.0),
                linear_color_stop(bg.opacity(0.0), 1.0),
            ));

        // Left side: the persistent participant indicator. Absolutely
        // positioned (full band height) so it sits independent of the
        // right-aligned model label's flex flow — adding/removing it never
        // shifts the model chrome. Its left edge is offset by
        // `INDICATOR_LEFT_PAD` (80px on macOS, 0 elsewhere) so the label clears
        // the macOS traffic lights instead of rendering behind them; `px_5`
        // then adds the same inner inset the prose column uses.
        if stage == OnboardingStage::Ready
            && let Some(indicator) = self.participant_indicator.as_ref()
        {
            let color = if indicator.is_error {
                theme.danger
            } else {
                theme.muted_foreground
            };
            band = band.child(
                div()
                    .absolute()
                    .left(INDICATOR_LEFT_PAD)
                    .px_5()
                    .text_sm()
                    .italic()
                    .text_color(color)
                    .child(SharedString::from(indicator.label)),
            );
        }

        if stage == OnboardingStage::Ready && self.model_revealed(cx) {
            band = band.child(
                div()
                    .id("model-label")
                    .px_5()
                    .text_sm()
                    .italic()
                    .text_color(theme.muted_foreground)
                    .cursor_pointer()
                    .hover(|s| s.text_color(theme.foreground))
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_model_picker(cx)))
                    .child(SharedString::from(self.current_model(cx))),
            );
        }
        if stage == OnboardingStage::Ready && self.model_picker_open {
            band = band.child(self.render_model_picker(cx));
        }
        band
    }

    /// The model picker: a quiet panel anchored under the title-bar band's
    /// right edge, listing `Core.models` with honest per-model info
    /// (context length, credits per token — straight from the `/models`
    /// payload). The current selection and the config default are marked;
    /// a small footer affordance persists the current selection as the
    /// default. Clicking outside dismisses; ⌥⌘M toggles.
    ///
    /// Hand-rolled rather than `gpui_component::Popover`: the open state
    /// must live on the view (⌥⌘M and behavior tests drive it without a
    /// pointer), the anchor is an absolutely-positioned band rather than an
    /// in-flow trigger, and the band already paints above the scroll
    /// content, so plain absolute positioning does everything `deferred(
    /// anchored(…))` would. `popover_style` keeps the surface itself
    /// consistent with gpui-component overlays under the Circadian theme.
    fn render_model_picker(&self, cx: &Context<Self>) -> gpui::Stateful<Div> {
        let theme = cx.theme();
        let current = self.current_model(cx);
        let default_model = self
            .config
            .read(cx)
            .state()
            .map(|s| s.default_model.clone())
            .unwrap_or_else(|| eidola_app_core::config::DEFAULT_MODEL.to_string());
        let models = self.models.read(cx).list().to_vec();

        let mut panel = v_flex()
            .id("model-picker")
            .occlude()
            .absolute()
            .top(TITLE_BAR_RESERVE)
            .right(px(12.))
            .w(px(340.))
            // Height cap so a long model list scrolls internally instead of
            // overflowing the window; the tracked handle lets keyboard
            // navigation scroll the highlighted row into view.
            .max_h(px(420.))
            .overflow_y_scroll()
            .track_scroll(&self.picker_scroll)
            .popover_style(cx)
            .py_1()
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.model_picker_open = false;
                cx.notify();
            }));

        if models.is_empty() {
            // Honest empty state: the model list hasn't loaded (or the
            // fetch failed) — say what a send will actually use.
            return panel.child(
                div()
                    .px_3()
                    .py_2()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(format!(
                        "Model list unavailable — sends use {current}."
                    ))),
            );
        }

        let highlighted = self.picker_highlighted;
        for (idx, model) in models.iter().enumerate() {
            let is_current = model.id == current;
            let is_default = model.id == default_model;
            let is_highlighted = highlighted == Some(idx);

            let mut markers: Vec<&str> = Vec::new();
            if is_current {
                markers.push("current");
            }
            if is_default {
                markers.push("default");
            }

            let mut name = div().text_sm().child(SharedString::from(model.id.clone()));
            if is_current {
                name = name.font_weight(gpui::FontWeight::SEMIBOLD);
            }
            let mut name_row = h_flex()
                .w_full()
                .justify_between()
                .items_baseline()
                .child(name);
            if !markers.is_empty() {
                name_row = name_row.child(
                    div()
                        .text_xs()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(markers.join(" · "))),
                );
            }

            let id = model.id.clone();
            panel = panel.child(
                v_flex()
                    .id(("model-row", idx))
                    .w_full()
                    .px_3()
                    .py_2()
                    .gap_0p5()
                    .when(idx > 0, |d| d.border_t_1().border_color(theme.border))
                    // Keyboard highlight: a quiet muted background, the same
                    // opacity the hover style uses so the two states feel
                    // consistent. The highlight is an honest state — it maps
                    // exactly to the row Enter would select.
                    .when(is_highlighted, |d| d.bg(theme.muted.opacity(0.5)))
                    .cursor_pointer()
                    .hover(|s| s.bg(theme.muted.opacity(0.5)))
                    .on_click(cx.listener(move |this, _, _, cx| this.select_model(id.clone(), cx)))
                    .child(name_row)
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(model_info_line(model))),
                    ),
            );
        }

        if current != default_model {
            // Quiet, secondary: persist this window's model as the config
            // default. Only offered when it would change anything.
            let label = format!("Set {current} as default");
            panel = panel.child(
                div()
                    .id("set-default-model")
                    .w_full()
                    .px_3()
                    .py_2()
                    .border_t_1()
                    .border_color(theme.border)
                    .text_xs()
                    .italic()
                    .text_color(theme.muted_foreground)
                    .cursor_pointer()
                    .hover(|s| s.text_color(theme.foreground))
                    .on_click(cx.listener(|this, _, _, cx| this.set_current_model_as_default(cx)))
                    .child(SharedString::from(label)),
            );
        }

        panel
    }
}

/// One honest line of per-model info for the picker, from the `/models`
/// payload: context length plus the credit rates that will actually be
/// charged. Per-request models show their flat rate; if the payload carried
/// no pricing at all, only the context length is shown — we don't invent
/// numbers.
fn model_info_line(model: &ModelInfo) -> String {
    // Per-request models (e.g. transcription) report no meaningful context
    // length; showing "0-token context" would be noise, not honesty.
    let ctx = (model.context_length > 0).then(|| {
        format!(
            "{}-token context",
            format_credits(model.context_length as i64)
        )
    });
    let price = if let Some(request) = model.request_credits {
        Some(format!("{} credits per request", format_rate(request)))
    } else if model.prompt_credits_per_token > 0.0 || model.completion_credits_per_token > 0.0 {
        Some(format!(
            "{} in / {} out credits per token",
            format_rate(model.prompt_credits_per_token),
            format_rate(model.completion_credits_per_token)
        ))
    } else {
        None
    };
    match (ctx, price) {
        (Some(ctx), Some(price)) => format!("{ctx} · {price}"),
        (Some(ctx), None) => ctx,
        (None, Some(price)) => price,
        (None, None) => "no published details".to_string(),
    }
}

/// Format a credit rate: whole thousands get separators ("9,000"),
/// everything else shows up to three decimals with trailing zeros trimmed
/// ("1.500" → "1.5", "0.530" → "0.53", "3.000" → "3").
fn format_rate(rate: f64) -> String {
    if rate >= 1000.0 && rate.fract() == 0.0 {
        return format_credits(rate as i64);
    }
    let s = format!("{rate:.3}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}
