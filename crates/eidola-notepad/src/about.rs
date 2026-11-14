//! About Eidola Notepad — a small singleton in the shape of Eidola's own.

use eidola_gui::probe::Probe as _;
use eidola_gui::space_view::TITLE_BAR_RESERVE;
use eidola_gui::theme;
use gpui::{
    Context, FocusHandle, InteractiveElement, IntoElement, ParentElement, Render, Styled, Window,
    div, px, relative, rems,
};
use gpui_component::{ActiveTheme, v_flex};

use crate::actions::CloseWindow;

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct AboutView {
    focus: FocusHandle,
}

impl AboutView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus = cx.focus_handle();
        focus.focus(window, cx);
        Self { focus }
    }
}

impl Render for AboutView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let version = format!("Version {VERSION}");

        let band = div()
            .id("about-drag")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(TITLE_BAR_RESERVE);

        eidola_gui::chrome::round_client_corners(v_flex(), window)
            .id("notepad-about")
            .track_focus(&self.focus)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| window.remove_window()))
            .relative()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .items_center()
            .justify_center()
            .gap_4()
            .px_8()
            .child(
                // The wordmark and purpose are the page's voice, so they take
                // the prose face; the version line is chrome.
                div()
                    .font_family(theme::FONT_FAMILY)
                    .text_size(px(28.))
                    .line_height(relative(1.2))
                    .child("Eidola Notepad"),
            )
            .child(div().w(rems(3.)).h(px(1.)).bg(theme.border))
            .child(
                div()
                    .id("about-version")
                    .probe_value(
                        "about/version",
                        gpui::Role::Label,
                        "Version",
                        version.clone(),
                    )
                    .text_sm()
                    .italic()
                    .text_color(theme.muted_foreground)
                    .child(version),
            )
            .child(
                div()
                    .font_family(theme::FONT_FAMILY)
                    .text_sm()
                    .text_center()
                    .text_color(theme.muted_foreground)
                    .child("Markdown files, written on Eidola’s writing surface."),
            )
            .child(eidola_gui::titlebar::make_draggable(
                band,
                "about-drag",
                window,
                cx,
            ))
    }
}
