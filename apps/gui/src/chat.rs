use eidola_app_core::SpaceMessage;
use gpui::{
    AppContext, Context, Entity, EventEmitter, FocusHandle, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Subscription,
    WeakEntity, Window, actions, div,
};
use gpui_component::{
    ActiveTheme, Disableable, IconName,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputState},
    label::Label,
    v_flex,
};

use crate::core::Core;

/// Default model to send to the inference endpoint.
const DEFAULT_MODEL: &str = "glm-5-1";

actions!(chat, [Send]);

pub struct ChatView {
    core: Entity<Core>,
    prompt_state: Entity<InputState>,
    space_id: Option<String>,
    /// Conversation history shown in the scroll view. `pub` so snapshot tests
    /// can render the view in a populated state without driving async chat.
    pub messages: Vec<SpaceMessage>,
    /// Whether to show the "Thinking…" indicator. `pub` for tests; production
    /// code only flips this from inside `submit`.
    pub thinking: bool,
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

    /// Test-only setter for snapshot tests.
    #[doc(hidden)]
    pub fn set_messages_for_test(&mut self, messages: Vec<SpaceMessage>) {
        self.messages = messages;
    }

    /// Test-only setter for snapshot tests.
    #[doc(hidden)]
    pub fn set_thinking_for_test(&mut self, thinking: bool) {
        self.thinking = thinking;
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

        // Kick off model loading so the chat can dispatch as soon as possible.
        core.update(cx, |core, cx| core.fetch_models(cx));

        Self {
            core,
            prompt_state,
            space_id: None,
            messages: Vec::new(),
            thinking: false,
            error: None,
            focus_handle,
            _subscriptions: Vec::new(),
        }
    }

    fn submit(&mut self, _: &Send, window: &mut Window, cx: &mut Context<Self>) {
        if self.thinking {
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

        // Clear the input and append the user message immediately.
        self.prompt_state.update(cx, |state, cx| {
            state.set_value("", window, cx);
        });
        self.messages.push(SpaceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        });
        self.thinking = true;
        self.error = None;
        cx.notify();

        let Some(app_core) = self.core.read(cx).app_core() else {
            // Stub core (behavior tests): the local state update above has
            // already happened; without a real backend there is nothing more
            // to do.
            return;
        };
        let space_id = self.space_id.clone();

        cx.spawn(async move |this: WeakEntity<Self>, cx| {
            let chat_rx = Core::chat(app_core.clone(), prompt, DEFAULT_MODEL.into(), space_id);
            let chat_outcome = chat_rx.await.unwrap_or_else(|_| {
                Err(eidola_app_core::error::AppError::Internal {
                    message: "chat task cancelled".into(),
                })
            });

            match chat_outcome {
                Ok(result) => {
                    let msgs_rx = Core::get_space_messages(app_core, result.space_id.clone());
                    let msgs = msgs_rx.await.unwrap_or_else(|_| {
                        Err(eidola_app_core::error::AppError::Internal {
                            message: "fetch messages task cancelled".into(),
                        })
                    });

                    let _ = this.update(cx, |this, cx| {
                        this.thinking = false;
                        this.space_id = Some(result.space_id);
                        match msgs {
                            Ok(messages) => this.messages = messages,
                            Err(e) => this.error = Some(e.to_string()),
                        }
                        cx.notify();
                    });
                }
                Err(e) => {
                    let _ = this.update(cx, |this, cx| {
                        this.thinking = false;
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

        let mut messages_col = v_flex().w_full().gap_0();
        for (idx, msg) in self.messages.iter().enumerate() {
            let bg = match msg.role.as_str() {
                "user" => theme.background,
                "error" => theme.danger.opacity(0.06),
                _ => theme.muted.opacity(0.4),
            };
            let fg = match msg.role.as_str() {
                "error" => theme.danger,
                _ => theme.foreground,
            };
            messages_col = messages_col.child(
                div()
                    .id(("msg", idx))
                    .w_full()
                    .px_5()
                    .py_3()
                    .bg(bg)
                    .text_color(fg)
                    .child(SharedString::from(msg.content.clone())),
            );
        }

        if self.thinking {
            messages_col = messages_col.child(
                h_flex()
                    .w_full()
                    .px_5()
                    .py_3()
                    .gap_2()
                    .bg(theme.muted.opacity(0.4))
                    .text_color(theme.muted_foreground)
                    .child(Label::new("Thinking…")),
            );
        }

        if let Some(err) = self.error.as_ref() {
            messages_col = messages_col.child(
                div()
                    .w_full()
                    .px_5()
                    .py_3()
                    .bg(theme.danger.opacity(0.06))
                    .text_color(theme.danger)
                    .child(SharedString::from(err.clone())),
            );
        }

        v_flex()
            .key_context("ChatView")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::submit))
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
                            .disabled(self.thinking)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.submit(&Send, window, cx);
                            })),
                    ),
            )
    }
}
