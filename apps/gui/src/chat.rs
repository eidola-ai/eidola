use std::sync::Arc;

use eidola_app_core::{ChatStreamEvent, SpaceMessage};
use gpui::{
    AppContext, Context, Div, Entity, EventEmitter, FocusHandle, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Subscription,
    WeakEntity, Window, actions, div, linear_color_stop, linear_gradient, px, relative, rems,
};
use gpui_component::{
    ActiveTheme, IconName,
    button::{Button, ButtonVariants},
    h_flex,
    highlighter::HighlightTheme,
    input::{Input, InputEvent, InputState},
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
            // `auto_grow` with a deliberately huge upper bound is the way we
            // disable the input's internal scrollbar: the bound caps `rows`
            // (which becomes the input's auto-height), so as long as the
            // user can't realistically type 10,000 wrapped lines in one
            // turn, the input keeps growing with content and the *outer*
            // scroll container handles overflow as one continuous unit.
            // This is the journal model — the editor isn't a fixed pane,
            // it's the page itself.
            InputState::new(window, cx)
                .multi_line(true)
                .auto_grow(1, 10_000)
                .placeholder("Begin writing…")
        });

        let focus_handle = cx.focus_handle();

        // Focus the input so the user can start typing immediately, like
        // opening a fresh journal page. The view's `focus_handle` is still
        // tracked on the root v_flex (behavior tests dispatch `Send`
        // through it), but production focus lives on the input itself —
        // which is the right cursor home for a "letter writing" feel.
        prompt_state.update(cx, |state, cx| state.focus(window, cx));

        // ⌘↩ on macOS lands inside the focused Input, which has its own
        // `secondary-enter` binding (gpui-component/.../input/state.rs:129):
        // it inserts a newline and emits `InputEvent::PressEnter { secondary
        // = true }`. The ChatView keybinding `cmd-enter → Send` therefore
        // never reaches us when the input has focus. Catch the event here
        // and trigger submit; `submit` trims the trailing newline before
        // forwarding the prompt upstream.
        let subscriptions = vec![cx.subscribe_in(
            &prompt_state,
            window,
            |this, _, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { secondary: true }) {
                    this.submit(&Send, window, cx);
                }
            },
        )];

        core.update(cx, |core, cx| core.fetch_models(cx));

        Self {
            core,
            prompt_state,
            space_id: None,
            messages: Vec::new(),
            streaming: None,
            error: None,
            focus_handle,
            _subscriptions: subscriptions,
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
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
        }

        // The composing input lives at the foot of the scroll, styled into
        // the same prose column the body uses, with no border/background
        // (`appearance(false)`) so it reads as a continuation of the page
        // rather than a separate chrome element. A "You" chapter delim
        // sits above it whenever there's preceding content, mirroring the
        // way earlier turns are introduced; on a fresh, empty page the
        // delim is omitted so the cursor sits cleanly at the top.
        //
        // The input grows freely with content via `auto_grow(1, 10_000)`
        // — the cap is large enough that no realistic prompt hits it, so
        // the input's internal scrollbar never engages and the outer
        // `overflow_y_scroll` div handles all of it (input + preceding
        // messages) as one continuous unit.
        let has_preceding =
            !self.messages.is_empty() || self.streaming.is_some() || self.error.is_some();
        if has_preceding {
            messages_col = messages_col.child(chapter_delim(
                participant_label("user"),
                theme.border,
                theme.muted_foreground,
            ));
        }
        messages_col = messages_col.child(
            div().w_full().px_5().child(
                prose_row().child(
                    prose().child(
                        Input::new(&self.prompt_state)
                            .appearance(false)
                            .text_size(PROSE_FONT_SIZE)
                            .line_height(relative(PROSE_LINE_HEIGHT))
                            .p_0()
                            .pb(composer_pb),
                    ),
                ),
            ),
        );

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
                    .w_full()
                    .flex_1()
                    .overflow_y_scroll()
                    .child(messages_col),
            )
            .child(title_bar_overlay(cx))
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
