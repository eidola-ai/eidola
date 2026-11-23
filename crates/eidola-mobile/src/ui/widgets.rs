//! The pieces every settings-like screen is built from, so they read as one
//! app: a header with its back chevron, rows that lead somewhere, pill
//! buttons, section labels, notes, a value with a Copy verb, and a text field.
//!
//! Every control is at least [`TOUCH`] tall, carries a probe (role, localized
//! label, stable name), and keeps the keyboard where it is unless it is a
//! field.

use gpui::{
    AnyElement, App, AppContext as _, ClickEvent, Context, Entity, FocusHandle, Focusable, Hsla,
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
use super::{GUTTER, HEADER_HEIGHT, TOUCH};

pub type OnClick = Box<dyn Fn(&ClickEvent, &mut Window, &mut App) + 'static>;

/// A tap on a control that is not a field must not take focus from one (the
/// keyboard would drop).
pub fn keep_keyboard(_: &gpui::MouseDownEvent, window: &mut Window, _: &mut App) {
    window.prevent_default();
}

/// A screen's header: the back chevron (when there is somewhere to go back
/// to), the title, and room for one trailing control.
pub fn header(
    name: &'static str,
    title: SharedString,
    back: Option<OnClick>,
    trailing: Option<AnyElement>,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let secondary = theme.secondary;
    div()
        .h(HEADER_HEIGHT)
        .flex_none()
        .flex()
        .items_center()
        .gap_1()
        .px(px(4.))
        .border_b_1()
        .border_color(theme.border)
        .map(|d| match back {
            Some(on_back) => d.child(
                div()
                    .id(SharedString::from(format!("{name}-back")))
                    .probe(format!("{name}/back"), Role::Button, msg::nav_back(cx))
                    .size(TOUCH)
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_full()
                    .active(move |s| s.bg(secondary))
                    .on_mouse_down(MouseButton::Left, keep_keyboard)
                    .on_click(on_back)
                    .child(Icon::new(IconName::ChevronLeft).size(px(26.))),
            ),
            None => d.child(div().w(px(16.)).flex_none()),
        })
        .child(
            div()
                .flex_1()
                .min_w_0()
                .truncate()
                .font_family(eidola_theme::FONT_FAMILY)
                .text_size(px(19.))
                .child(title),
        )
        .child(match trailing {
            Some(el) => el,
            None => div().size(TOUCH).flex_none().into_any_element(),
        })
}

/// A row that leads somewhere: a label, an optional detail beneath it, and a
/// chevron.
pub fn nav_row(
    name: &'static str,
    label: SharedString,
    detail: Option<SharedString>,
    detail_color: Option<Hsla>,
    on_click: OnClick,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let secondary = theme.secondary;
    let muted = theme.muted_foreground;
    div()
        .id(name)
        .probe(name, Role::Button, label.clone())
        .w_full()
        .min_h(px(56.))
        .px(GUTTER)
        .py(px(10.))
        .flex()
        .items_center()
        .gap_3()
        .border_b_1()
        .border_color(theme.border)
        .active(move |s| s.bg(secondary))
        .on_mouse_down(MouseButton::Left, keep_keyboard)
        .on_click(on_click)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(px(2.))
                .child(
                    div()
                        .font_family(eidola_theme::FONT_FAMILY)
                        .text_size(px(18.))
                        .child(label),
                )
                .children(detail.map(|d| {
                    div()
                        .text_size(px(14.))
                        .text_color(detail_color.unwrap_or(muted))
                        .child(d)
                })),
        )
        .child(
            Icon::new(IconName::ChevronRight)
                .size(px(18.))
                .text_color(muted),
        )
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pill {
    Primary,
    Secondary,
    Danger,
}

/// A pill button. A disabled one paints its label and takes no tap.
pub fn pill(
    name: impl Into<SharedString>,
    label: SharedString,
    kind: Pill,
    enabled: bool,
    on_click: OnClick,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let name = name.into();
    let (bg, fg) = match kind {
        Pill::Primary => (theme.primary, theme.primary_foreground),
        Pill::Secondary => (theme.secondary, theme.secondary_foreground),
        Pill::Danger => (theme.danger, theme.danger_foreground),
    };
    div()
        .id(name.clone())
        .probe(name, Role::Button, label.clone())
        .min_h(TOUCH)
        .px(px(20.))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .bg(bg)
        .text_color(fg)
        .text_size(px(16.))
        .text_center()
        .when(!enabled, |d| d.opacity(0.45))
        .on_mouse_down(MouseButton::Left, keep_keyboard)
        .when(enabled, |d| d.on_click(on_click))
        .child(label)
}

/// A small heading over a group of rows.
pub fn section(label: SharedString, cx: &App) -> impl IntoElement {
    div()
        .px(GUTTER)
        .pt(px(22.))
        .pb(px(6.))
        .text_size(px(13.))
        .text_color(cx.theme().muted_foreground)
        .child(label)
}

/// A paragraph of quiet explanation.
pub fn note(text: SharedString, cx: &App) -> impl IntoElement {
    div()
        .px(GUTTER)
        .py(px(6.))
        .text_size(px(14.))
        .line_height(px(20.))
        .text_color(cx.theme().muted_foreground)
        .child(text)
}

/// A refusal, said where it happened.
pub fn error_note(name: impl Into<SharedString>, text: SharedString, cx: &App) -> impl IntoElement {
    let name = name.into();
    div()
        .id(name.clone())
        .probe(name, Role::Alert, text.clone())
        .mx(GUTTER)
        .my(px(6.))
        .px(px(12.))
        .py(px(10.))
        .rounded_lg()
        .border_1()
        .border_color(cx.theme().danger)
        .text_size(px(14.))
        .line_height(px(20.))
        .text_color(cx.theme().danger)
        .child(text)
}

/// A labelled value with a Copy verb (an account id, a secret, a digest).
pub fn copy_row(
    key: &'static str,
    label: SharedString,
    value: SharedString,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let to_copy = value.to_string();
    let copy_label = msg::nav_copy_label(cx, label.to_string());
    div()
        .px(GUTTER)
        .py(px(8.))
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(13.))
                .text_color(theme.muted_foreground)
                .child(label.clone()),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    div()
                        .id(SharedString::from(format!("{key}-value")))
                        .probe(format!("{key}/value"), Role::Label, value.clone())
                        .flex_1()
                        .min_w_0()
                        .font_family(theme::MONO_FAMILY)
                        .text_size(px(14.))
                        .line_height(px(20.))
                        .child(value),
                )
                .child(
                    div()
                        .id(SharedString::from(format!("{key}-copy")))
                        .probe(format!("{key}/copy"), Role::Button, copy_label)
                        .flex_none()
                        .min_h(TOUCH)
                        .px(px(14.))
                        .flex()
                        .items_center()
                        .rounded_full()
                        .bg(theme.secondary)
                        .text_color(theme.secondary_foreground)
                        .text_size(px(15.))
                        .on_mouse_down(MouseButton::Left, keep_keyboard)
                        .on_click(move |_, _, cx| {
                            cx.write_to_clipboard(gpui::ClipboardItem::new_string(to_copy.clone()))
                        })
                        .child(msg::nav_copy(cx)),
                ),
        )
}

/// A labelled line of text (a figure, a version).
pub fn value_row(
    name: &'static str,
    label: SharedString,
    value: SharedString,
    mono: bool,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .id(name)
        .probe(name, Role::Label, format!("{label}: {value}"))
        .px(GUTTER)
        .py(px(8.))
        .flex()
        .flex_col()
        .gap(px(2.))
        .child(
            div()
                .text_size(px(13.))
                .text_color(theme.muted_foreground)
                .child(label),
        )
        .child(
            div()
                .when(mono, |d| {
                    d.font_family(theme::MONO_FAMILY).text_size(px(14.))
                })
                .when(!mono, |d| {
                    d.font_family(eidola_theme::FONT_FAMILY).text_size(px(18.))
                })
                .line_height(px(22.))
                .child(value),
        )
}

/// A digest in four groups of sixteen, as the shell's sheet shows it.
pub fn grouped_digest(digest: &str) -> SharedString {
    digest
        .as_bytes()
        .chunks(16)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
        .into()
}

/// A one-line text field on the markdown editor (the composer's field, so
/// the system keyboard and its input methods behave the same way): the touch
/// that lands in it asks for the keyboard.
pub struct TextField {
    editor: Entity<MarkdownEditorState>,
    _editor: Subscription,
}

impl TextField {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let editor = cx.new(|cx| MarkdownEditorState::new(window, cx));
        let subscription = cx.subscribe(&editor, |_, _, _: &MarkdownEditorEvent, cx| cx.notify());
        Self {
            editor,
            _editor: subscription,
        }
    }

    /// The text, with any line breaks and surrounding space a paste or the
    /// return key left in a one-line value taken out.
    pub fn value(&self, cx: &App) -> String {
        let editor = self.editor.read(cx);
        editor
            .value()
            .split(['\n', '\r'])
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("")
    }

    /// End an open composition, so the value read next is what is shown.
    pub fn commit(&mut self, cx: &mut Context<Self>) {
        self.editor.update(cx, |e, cx| {
            if e.is_composing() {
                e.end_composition(cx);
            }
        });
    }

    pub fn clear(&mut self, cx: &mut Context<Self>) {
        self.editor.update(cx, |e, cx| e.set_value("", cx));
    }
}

impl Focusable for TextField {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.editor.focus_handle(cx)
    }
}

/// How a [`TextField`] is drawn: its probe, its accessible name and its
/// placeholder are the host's.
pub fn text_field(
    field: &Entity<TextField>,
    name: &'static str,
    label: SharedString,
    placeholder: SharedString,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let state = field.read(cx);
    let empty = state.editor.read(cx).value().is_empty() && !state.editor.read(cx).is_composing();
    let mut style = theme::prose_style(cx);
    style.font_family = theme::MONO_FAMILY.into();
    style.font_size = px(16.);
    div()
        .mx(GUTTER)
        .my(px(4.))
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(13.))
                .text_color(theme.muted_foreground)
                .child(label.clone()),
        )
        .child(
            div()
                .id(name)
                .probe(name, Role::TextInput, label)
                .relative()
                .min_h(TOUCH)
                .px(px(14.))
                .py(px(10.))
                .rounded(px(12.))
                .border_1()
                .border_color(theme.border)
                .bg(theme.background)
                .capture_any_mouse_down(|_, window, _| window.request_virtual_keyboard())
                .child(MarkdownEditor::new(&state.editor).style(style))
                .when(empty, |d| {
                    d.child(
                        div()
                            .absolute()
                            .top(px(10.))
                            .left(px(14.))
                            .right(px(14.))
                            .font_family(theme::MONO_FAMILY)
                            .text_size(px(16.))
                            .text_color(theme.muted_foreground)
                            .child(placeholder),
                    )
                }),
        )
}

impl Render for TextField {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
    }
}

/// A read-only block of markdown in the prose style (an onboarding page's
/// words). The text is the host's to push each render, so a locale change
/// reads in the new language.
pub struct Prose {
    editor: Entity<MarkdownEditorState>,
    shown: SharedString,
}

impl Prose {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        Self {
            editor: cx.new(|cx| MarkdownEditorState::new(window, cx)),
            shown: SharedString::default(),
        }
    }

    /// Show `text`, if it is not already what is shown.
    pub fn sync(&mut self, text: SharedString, cx: &mut Context<Self>) {
        if self.shown != text {
            self.shown = text.clone();
            self.editor
                .update(cx, |e, cx| e.set_value(text.to_string(), cx));
        }
    }
}

impl Render for Prose {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div().px(GUTTER).child(
            MarkdownEditor::new(&self.editor)
                .style(theme::prose_style(cx))
                .disabled(true),
        )
    }
}
