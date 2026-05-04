use std::sync::Arc;

use eidola_app_core::{ChatStreamEvent, SpaceMessage};
use gpui::{
    AppContext, Context, Entity, EventEmitter, FocusHandle, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Subscription,
    WeakEntity, Window, actions, div, linear_color_stop, linear_gradient,
};
use gpui_component::{
    ActiveTheme, Disableable, IconName,
    button::{Button, ButtonVariants},
    h_flex,
    highlighter::HighlightTheme,
    input::{Input, InputState},
    label::Label,
    text::{TextView, TextViewStyle},
    v_flex,
};

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

pub struct ChatView {
    core: Entity<Core>,
    prompt_state: Entity<InputState>,
    space_id: Option<String>,
    /// Conversation history shown in the scroll view. `pub` so snapshot tests
    /// can render the view in a populated state without driving async chat.
    pub messages: Vec<ChatMessageView>,
    /// In-flight streaming assistant response, or `None` when idle.
    pub streaming: Option<StreamingResponse>,
    error: Option<String>,

    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl ChatView {
    /// The focus handle the view tracks. Exposed so behavior tests can dispatch
    /// actions through it the same way real keystrokes would.
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    /// Test-only access to the prompt input state, for behavior tests that
    /// want to populate it the way a typing user would.
    #[doc(hidden)]
    pub fn prompt_state_for_test(&self) -> Entity<InputState> {
        self.prompt_state.clone()
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
}

impl ChatView {
    pub fn new(core: Entity<Core>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let prompt_state = cx.new(|cx| {
            InputState::new(window, cx)
                .multi_line(true)
                .auto_grow(1, 8)
                .placeholder("Ask anything…")
        });

        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        core.update(cx, |core, cx| core.fetch_models(cx));

        Self {
            core,
            prompt_state,
            space_id: None,
            messages: Vec::new(),
            streaming: None,
            error: None,
            focus_handle,
            _subscriptions: Vec::new(),
        }
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
                    reasoning_expanded: same_position
                        .then(|| prior.is_some_and(|p| p.reasoning_expanded))
                        .unwrap_or(false),
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
            .prompt_state
            .read(cx)
            .value()
            .to_string()
            .trim()
            .to_string();
        if prompt.is_empty() {
            return;
        }

        self.prompt_state.update(cx, |state, cx| {
            state.set_value("", window, cx);
        });
        self.messages.push(ChatMessageView::new(SpaceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        }));
        self.streaming = Some(StreamingResponse::default());
        self.error = None;
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
                        this.streaming = None;
                        this.error = Some(e.to_string());
                        cx.notify();
                    });
                }
            }
        })
        .detach();
    }
}

impl EventEmitter<()> for ChatView {}

impl Render for ChatView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let markdown_style = markdown_style(theme.mode.is_dark());

        let mut messages_col = v_flex().w_full().gap_0().pt(TITLE_BAR_RESERVE);
        for (idx, entry) in self.messages.iter().enumerate() {
            let msg = &entry.message;
            let bg = match msg.role.as_str() {
                "user" => theme.background,
                "error" => theme.danger.opacity(0.06),
                _ => theme.muted.opacity(0.4),
            };
            let fg = match msg.role.as_str() {
                "error" => theme.danger,
                _ => theme.foreground,
            };

            let mut row = v_flex()
                .id(("msg-row", idx))
                .w_full()
                .px_5()
                .py_3()
                .gap_2()
                .bg(bg)
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
                // label ends up center-aligned.
                row = row.child(
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
                );
                if entry.reasoning_expanded {
                    row = row.child(
                        div().pl_4().text_color(theme.muted_foreground).child(
                            TextView::markdown(("thinking-body", idx), reasoning.to_string())
                                .selectable(true)
                                .style(markdown_style.clone()),
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
            row = row.child(body);

            messages_col = messages_col.child(row);
        }

        if let Some(s) = self.streaming.as_ref() {
            let bg = theme.muted.opacity(0.4);
            let fg = theme.foreground;
            let muted_fg = theme.muted_foreground;
            let danger = theme.danger;

            let mut col = v_flex()
                .id("streaming-row")
                .w_full()
                .px_5()
                .py_3()
                .gap_2()
                .bg(bg)
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
                );
                if s.expanded {
                    col = col.child(
                        div().pl_4().text_color(muted_fg).child(
                            TextView::markdown(
                                ("streaming-thinking-body", 0usize),
                                s.reasoning.clone(),
                            )
                            .selectable(true)
                            .style(markdown_style.clone()),
                        ),
                    );
                }
            } else if s.content.is_empty() {
                // No reasoning *and* no content yet — show the "still
                // working" status indicator. Plain Label, no toggle, no
                // markdown plumbing.
                col = col.child(Label::new(SharedString::from("Thinking…")).text_color(muted_fg));
            }

            if !s.content.is_empty() {
                col = col.child(
                    TextView::markdown(("streaming-body", 0usize), s.content.clone())
                        .selectable(true)
                        .style(markdown_style.clone()),
                );
            }

            if let Some(err) = s.error.as_deref() {
                col = col.child(
                    div().text_color(danger).child(
                        TextView::markdown(("streaming-error", 0usize), err.to_string())
                            .selectable(true)
                            .style(markdown_style.clone()),
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
            // plain text.
            messages_col = messages_col.child(
                div()
                    .w_full()
                    .px_5()
                    .py_3()
                    .bg(theme.danger.opacity(0.06))
                    .text_color(theme.danger)
                    .child(
                        TextView::markdown(("chat-error", 0usize), err.clone())
                            .selectable(true)
                            .style(markdown_style.clone()),
                    ),
            );
        }

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
            .child(
                div()
                    .id("scroll")
                    .flex_1()
                    .overflow_y_scroll()
                    .child(messages_col),
            )
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .px_4()
                    .py_3()
                    .border_t_1()
                    .border_color(theme.border)
                    .child(div().flex_1().child(Input::new(&self.prompt_state)))
                    .child(
                        Button::new("send")
                            .primary()
                            .icon(IconName::ArrowUp)
                            .disabled(self.streaming.is_some())
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.submit(&Send, window, cx);
                            })),
                    ),
            )
            .child(title_bar_overlay(cx))
    }
}

/// `TextViewStyle` for chat message bodies. The `is_dark` flag and matching
/// `HighlightTheme` are wired off the active Circadian mode so fenced code
/// blocks render against a backdrop that matches the rest of the chrome.
fn markdown_style(is_dark: bool) -> TextViewStyle {
    let highlight = if is_dark {
        HighlightTheme::default_dark().clone()
    } else {
        HighlightTheme::default_light().clone()
    };
    TextViewStyle {
        is_dark,
        highlight_theme: highlight,
        ..TextViewStyle::default()
    }
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
