//! The composer: a multi-line field set in the prose style, so what is typed
//! reads exactly as the post it becomes, and a send button.
//!
//! The field is the desktop app's markdown editor, enabled. On a phone two
//! things are the host's: **asking for the software keyboard** — gpui leaves
//! that to the gesture that focuses a field ([`Window::request_virtual_keyboard`]),
//! so the field asks for it on the touch that lands in it — and **IME**,
//! which needs nothing here: the editor is gpui's input handler, and the web
//! backend mirrors it into a hidden text area the system keyboard and its
//! composition (Chinese, Japanese, dictation) type into. A send ends any
//! composition still open first, so the words sent are the words shown.
//!
//! The composer sits at the bottom of the visible area, which is the bottom of
//! the screen with the keyboard down and the top of the keyboard with it up.

use gpui::{
    AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable,
    InteractiveElement as _, IntoElement, MouseButton, ParentElement as _, Render, Role,
    SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window, div,
    prelude::FluentBuilder as _, px,
};
use gpui_component::{ActiveTheme as _, Icon};
use gpui_kit_assets::IconName;
use gpui_markdown_editor::{MarkdownEditor, MarkdownEditorEvent, MarkdownEditorState};

use super::i18n::msg;
use super::probe::Probe as _;
use super::theme;
use super::{GUTTER, TOUCH};

/// The most of the screen the field grows to before it scrolls.
const FIELD_MAX_HEIGHT: gpui::Pixels = px(168.);

/// A post the next send replies to.
#[derive(Clone, Debug)]
pub struct ReplyTarget {
    pub action_id: String,
    pub author: SharedString,
}

#[derive(Clone, Debug)]
pub enum ComposerEvent {
    Send {
        text: String,
        reply_to: Option<ReplyTarget>,
    },
    /// The field gained focus (the conversation scrolls to its end).
    Focused,
    /// The field became empty (sent, or cleared): free for a refused post's
    /// words.
    Emptied,
}

pub struct Composer {
    editor: Entity<MarkdownEditorState>,
    reply_to: Option<ReplyTarget>,
    /// A save is in flight: the send button rests until it answers.
    holding: bool,
    _editor: Subscription,
}

impl EventEmitter<ComposerEvent> for Composer {}

impl Focusable for Composer {
    fn focus_handle(&self, cx: &gpui::App) -> FocusHandle {
        self.editor.focus_handle(cx)
    }
}

impl Composer {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let editor = cx.new(|cx| MarkdownEditorState::new(window, cx));
        let subscription = cx.subscribe(&editor, |this, _, event: &MarkdownEditorEvent, cx| {
            match event {
                MarkdownEditorEvent::Focus => cx.emit(ComposerEvent::Focused),
                MarkdownEditorEvent::Change if this.is_empty(cx) => cx.emit(ComposerEvent::Emptied),
                _ => {}
            }
            cx.notify();
        });
        Self {
            editor,
            reply_to: None,
            holding: false,
            _editor: subscription,
        }
    }

    pub fn reply_to(&self) -> Option<&ReplyTarget> {
        self.reply_to.as_ref()
    }

    /// Write a reply to `target` (or, with `None`, to the conversation's end).
    pub fn set_reply(&mut self, target: Option<ReplyTarget>, cx: &mut Context<Self>) {
        self.reply_to = target;
        cx.notify();
    }

    pub fn set_holding(&mut self, holding: bool, cx: &mut Context<Self>) {
        self.holding = holding;
        cx.notify();
    }

    /// The send was accepted: the field empties, and the reply is spent.
    pub fn sent(&mut self, cx: &mut Context<Self>) {
        self.editor.update(cx, |e, cx| e.set_value("", cx));
        self.reply_to = None;
        cx.notify();
    }

    /// What the field holds.
    pub fn text(&self, cx: &gpui::App) -> String {
        self.editor.read(cx).value().to_string()
    }

    /// Nothing written in the field, and no composition open.
    pub fn is_empty(&self, cx: &gpui::App) -> bool {
        let editor = self.editor.read(cx);
        editor.value().trim().is_empty() && !editor.is_composing()
    }

    /// A refused save's words come back into the field. The caller gives
    /// them only to an empty field ([`Self::is_empty`]), so nothing the
    /// reader has started writing is replaced.
    pub fn restore(&mut self, text: &str, reply_to: Option<ReplyTarget>, cx: &mut Context<Self>) {
        self.editor
            .update(cx, |e, cx| e.set_value(text.to_string(), cx));
        self.reply_to = reply_to;
        cx.notify();
    }

    pub fn focus(&self, window: &mut Window, cx: &mut Context<Self>) {
        self.editor.read(cx).focus_handle(cx).focus(window, cx);
    }

    fn can_send(&self, cx: &gpui::App) -> bool {
        !self.holding && !self.editor.read(cx).value().trim().is_empty()
    }

    fn send(&mut self, cx: &mut Context<Self>) {
        // Sending ends an open composition, so what is sent is what is shown.
        self.editor.update(cx, |e, cx| {
            if e.is_composing() {
                e.end_composition(cx);
            }
        });
        if !self.can_send(cx) {
            return;
        }
        let text = self.editor.read(cx).value().to_string();
        cx.emit(ComposerEvent::Send {
            text,
            reply_to: self.reply_to.clone(),
        });
    }
}

impl Render for Composer {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let empty = self.editor.read(cx).value().is_empty() && !self.editor.read(cx).is_composing();
        let can_send = self.can_send(cx);
        let style = theme::prose_style(cx);

        let reply = self.reply_to.clone().map(|target| {
            div()
                .flex()
                .items_center()
                .gap_2()
                .pl(px(4.))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .truncate()
                        .text_size(px(14.))
                        .text_color(theme.muted_foreground)
                        .child(msg::conversation_replying_to(cx, target.author.to_string())),
                )
                .child(
                    div()
                        .id("composer-cancel-reply")
                        .probe(
                            "composer/cancel-reply",
                            Role::Button,
                            msg::conversation_cancel_reply(cx),
                        )
                        .size(TOUCH)
                        .flex()
                        .items_center()
                        .justify_center()
                        .on_mouse_down(MouseButton::Left, |_, window, _| window.prevent_default())
                        .on_click(cx.listener(|this, _, _, cx| this.set_reply(None, cx)))
                        .child(
                            Icon::new(IconName::Close)
                                .size(px(18.))
                                .text_color(theme.muted_foreground),
                        ),
                )
        });

        let field = div()
            .id("composer-field")
            .probe("composer/field", Role::TextInput, msg::composer_label(cx))
            .flex_1()
            .min_w_0()
            .min_h(TOUCH)
            .max_h(FIELD_MAX_HEIGHT)
            .overflow_y_scroll()
            .px(px(14.))
            .py(px(6.))
            .rounded(px(22.))
            .border_1()
            .border_color(theme.border)
            .bg(theme.background)
            .relative()
            // The touch that lands in the field asks for the keyboard: on a
            // phone a focus alone does not summon it. Captured, so it runs
            // before the editor takes the press.
            .capture_any_mouse_down(|_, window, _| window.request_virtual_keyboard())
            .child(MarkdownEditor::new(&self.editor).style(style))
            .when(empty, |d| {
                d.child(
                    div()
                        .absolute()
                        .top(px(6.))
                        .left(px(14.))
                        .right(px(14.))
                        .font_family(eidola_theme::FONT_FAMILY)
                        .text_size(eidola_theme::prose::PROSE_FONT_SIZE * theme::font_scale(cx))
                        .line_height(gpui::rems(eidola_theme::prose::PROSE_LINE_HEIGHT))
                        .text_color(theme.muted_foreground)
                        .child(msg::composer_placeholder(cx)),
                )
            });

        let send = div()
            .id("composer-send")
            .probe("composer/send", Role::Button, msg::composer_send(cx))
            .flex_none()
            .size(TOUCH)
            .flex()
            .items_center()
            .justify_center()
            .rounded_full()
            .bg(if can_send {
                theme.primary
            } else {
                theme.secondary
            })
            // Tapping send must not take focus from the field (that would
            // drop the keyboard between two messages).
            .on_mouse_down(MouseButton::Left, |_, window, _| window.prevent_default())
            .on_click(cx.listener(|this, _, _, cx| this.send(cx)))
            .child(
                Icon::new(IconName::ArrowUp)
                    .size(px(22.))
                    .text_color(if can_send {
                        theme.primary_foreground
                    } else {
                        theme.muted_foreground
                    }),
            );

        div()
            .flex_none()
            .flex()
            .flex_col()
            .gap_1()
            .px(px(12.))
            .pt(px(6.))
            .pb(px(8.))
            .border_t_1()
            .border_color(theme.border)
            .bg(theme.background)
            .children(reply)
            .child(
                div()
                    .flex()
                    .items_end()
                    .gap_2()
                    .pl(GUTTER - px(12.))
                    .child(field)
                    .child(send),
            )
    }
}
