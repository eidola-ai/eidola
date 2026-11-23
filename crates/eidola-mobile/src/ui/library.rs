//! The Library, phone-style: the reader's conversations, newest activity
//! first, as a table of contents — a title in the prose face, and when it was
//! last active. A tap opens one; the pen in the corner starts a new one.

use eidola_app_core::SpaceInfo;
use gpui::{
    Context, EventEmitter, InteractiveElement as _, IntoElement, ListAlignment, ListState,
    MouseButton, ParentElement as _, Render, Role, SharedString, StatefulInteractiveElement as _,
    Styled as _, Window, div, list, prelude::FluentBuilder as _, px,
};
use gpui_component::{ActiveTheme as _, Icon};
use gpui_kit_assets::IconName;

use super::i18n::msg;
use super::loadable::Loadable;
use super::probe::Probe as _;
use super::stores::Stores;
use super::{GUTTER, HEADER_HEIGHT, TOUCH, fmt_when};

#[derive(Clone, Debug)]
pub enum LibraryEvent {
    Open(String),
    New,
}

pub struct LibraryView {
    stores: Stores,
    list: ListState,
    /// The ids of the rows the list holds, in order: a re-list that changes
    /// them resets the list's measurements.
    ids: Vec<String>,
}

impl EventEmitter<LibraryEvent> for LibraryView {}

impl LibraryView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.spaces, |_, _, cx| cx.notify()).detach();
        Self {
            stores,
            list: ListState::new(0, ListAlignment::Top, px(800.)),
            ids: Vec::new(),
        }
    }

    fn sync_rows(&mut self, rows: &[SpaceInfo]) {
        let ids: Vec<String> = rows.iter().map(|s| s.id.clone()).collect();
        if ids != self.ids {
            self.list.reset(ids.len());
            self.ids = ids;
        }
    }

    fn render_row(&mut self, ix: usize, cx: &mut Context<Self>) -> gpui::AnyElement {
        let Some(info) = self
            .stores
            .spaces
            .read(cx)
            .index()
            .value()
            .and_then(|rows| rows.get(ix))
            .cloned()
        else {
            return div().into_any_element();
        };
        let theme = cx.theme();
        let title = row_title(&info, cx);
        let when = fmt_when(info.last_activity_at);
        // The first words, under a title that is not them already.
        let snippet = info
            .title
            .as_ref()
            .and(info.snippet.clone())
            .filter(|s| Some(s.trim()) != info.title.as_deref().map(str::trim))
            .map(SharedString::from);
        let id = info.id.clone();
        div()
            .id(("library-row", ix))
            .probe(
                format!("library/row/{ix}"),
                Role::Button,
                msg::library_row_label(cx, title.to_string(), when.to_string()),
            )
            .w_full()
            .min_h(px(72.))
            .px(GUTTER)
            .py(px(12.))
            .flex()
            .flex_col()
            .justify_center()
            .gap(px(2.))
            .border_b_1()
            .border_color(theme.border)
            .active(|s| s.bg(theme.secondary))
            .on_click(cx.listener(move |_, _, _, cx| cx.emit(LibraryEvent::Open(id.clone()))))
            .child(
                div()
                    .flex()
                    .items_baseline()
                    .gap_3()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .font_family(eidola_theme::FONT_FAMILY)
                            .text_size(px(19.))
                            .line_height(px(26.))
                            .truncate()
                            .child(title),
                    )
                    .child(
                        div()
                            .flex_none()
                            .text_size(px(13.))
                            .text_color(theme.muted_foreground)
                            .child(when),
                    ),
            )
            .children(snippet.map(|s| {
                div()
                    .text_size(px(14.))
                    .text_color(theme.muted_foreground)
                    .truncate()
                    .child(s)
            }))
            .into_any_element()
    }
}

fn row_title(info: &SpaceInfo, cx: &gpui::App) -> SharedString {
    info.title
        .clone()
        .or_else(|| info.snippet.clone())
        .map(SharedString::from)
        .unwrap_or_else(|| msg::library_untitled(cx))
}

impl Render for LibraryView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let index = self.stores.spaces.read(cx).index().clone();
        let theme = cx.theme().clone();
        if let Some(rows) = index.value() {
            self.sync_rows(rows);
        }

        let header = div()
            .h(HEADER_HEIGHT)
            .flex_none()
            .flex()
            .items_center()
            .justify_between()
            .pl(GUTTER)
            .pr(px(8.))
            .child(
                div()
                    .font_family(eidola_theme::FONT_FAMILY)
                    .text_size(px(28.))
                    .child(msg::library_title(cx)),
            )
            .child(
                div()
                    .id("library-new")
                    .probe("library/new", Role::Button, msg::library_new(cx))
                    .size(TOUCH)
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_full()
                    .active(|s| s.bg(theme.secondary))
                    .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(LibraryEvent::New)))
                    .child(
                        Icon::new(IconName::SquarePen)
                            .size(px(22.))
                            .text_color(theme.foreground),
                    ),
            );

        let body = match &index {
            Loadable::NotLoaded | Loadable::Loading => div().flex_1().into_any_element(),
            Loadable::Failed { prior: None, .. } => centered(
                msg::library_load_failed(cx),
                Some((
                    "library/retry",
                    msg::library_retry(cx),
                    cx.listener(|this, _, _, cx| {
                        this.stores.spaces.update(cx, |s, cx| s.refresh(cx));
                    }),
                )),
                &theme,
            ),
            _ if self.ids.is_empty() => centered(
                msg::library_empty(cx),
                Some((
                    "library/start",
                    msg::library_empty_start(cx),
                    cx.listener(|_, _, _, cx| cx.emit(LibraryEvent::New)),
                )),
                &theme,
            ),
            _ => list(
                self.list.clone(),
                cx.processor(|this, ix, _, cx| this.render_row(ix, cx)),
            )
            .flex_1()
            .size_full()
            .into_any_element(),
        };

        // A refresh that failed over rows still shown: they are dated, so say
        // so, with the way to ask again.
        let stale = matches!(&index, Loadable::Failed { prior: Some(_), .. });

        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header)
            .when(stale, |d| {
                d.child(
                    div()
                        .id("library-refresh")
                        .probe(
                            "library/refresh-retry",
                            Role::Button,
                            msg::library_refresh_failed(cx),
                        )
                        .mx(GUTTER)
                        .min_h(TOUCH)
                        .flex()
                        .items_center()
                        .justify_center()
                        .rounded_lg()
                        .bg(theme.secondary)
                        .text_size(px(14.))
                        .text_color(theme.secondary_foreground)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.stores.spaces.update(cx, |s, cx| s.refresh(cx));
                        }))
                        .child(msg::library_refresh_failed(cx)),
                )
            })
            .child(div().flex_1().min_h_0().flex().flex_col().child(body))
    }
}

/// A centred line with an optional verb beneath it.
fn centered(
    line: SharedString,
    verb: Option<(
        &'static str,
        SharedString,
        impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
    )>,
    theme: &gpui_component::Theme,
) -> gpui::AnyElement {
    div()
        .flex_1()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap_4()
        .px(GUTTER)
        .child(
            div()
                .font_family(eidola_theme::FONT_FAMILY)
                .text_size(px(19.))
                .text_color(theme.muted_foreground)
                .text_center()
                .child(line),
        )
        .children(verb.map(|(name, label, on_click)| {
            div()
                .id(name)
                .probe(name, Role::Button, label.clone())
                .min_h(TOUCH)
                .px(px(20.))
                .flex()
                .items_center()
                .rounded_full()
                .bg(theme.primary)
                .text_color(theme.primary_foreground)
                .text_size(px(16.))
                .on_click(on_click)
                .child(label)
        }))
        .into_any_element()
}
