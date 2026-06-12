//! About window — a small singleton showing the wordmark, version, a quiet
//! line of purpose, source note, and a "View on GitHub" link.
//!
//! Same transparent-titlebar treatment as all other windows. Singleton so
//! repeated "About Eidola" invocations raise the existing window rather than
//! stacking new ones. ~360×420 px.

use gpui::{
    Context, FocusHandle, InteractiveElement, IntoElement, ParentElement, Render, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px, relative, rems,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use crate::actions::CloseWindow;

/// The version string baked in at compile time.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// GitHub repository URL shown as the "View on GitHub" link.
const REPO_URL: &str = "https://github.com/eidola-ai/eidola";

/// Vertical reserve for the macOS traffic lights (same pattern as all
/// other windows with `transparent_titlebar`).
#[cfg(target_os = "macos")]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(36.);
#[cfg(not(target_os = "macos"))]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(0.);

pub struct AboutView {
    focus_handle: FocusHandle,
}

impl AboutView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);
        Self { focus_handle }
    }

    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for AboutView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        // Wordmark block: large "Eidola" + a hairline rule underneath,
        // matching the welcome page's title-page treatment.
        let wordmark = v_flex()
            .gap_3()
            .items_center()
            .child(
                div()
                    .text_size(px(32.))
                    .line_height(relative(1.2))
                    .child("Eidola"),
            )
            .child(div().w(rems(3.)).h(px(1.)).bg(theme.border));

        // Version line: muted, italic, small — unobtrusive.
        let version = div()
            .text_sm()
            .italic()
            .text_color(theme.muted_foreground)
            .child(SharedString::from(format!("v{VERSION}")));

        // Purpose copy: echoes the welcome page's voice (same three-sentence
        // set minus the call to action — the reader has already begun).
        let purpose = v_flex()
            .gap_3()
            .text_sm()
            .text_color(theme.muted_foreground)
            .child(
                "A quiet page for thinking with a machine — private by \
                 construction, not by policy.",
            )
            .child(
                "Every request runs inside sealed, hardware-attested enclaves, \
                 and this app verifies the cryptographic evidence before a word \
                 leaves your machine.",
            );

        // Source note. Deliberately no license claim: the repository does
        // not yet carry a LICENSE file, and the About page must not assert
        // terms that aren't durably true (the no-fake-states rule applies
        // to legal claims too). Add the real license line when one lands.
        let license = div()
            .text_xs()
            .text_color(theme.muted_foreground)
            .child("Source available on GitHub.");

        // "View on GitHub" link — `cx.open_url` opens the default browser.
        let github_link = div()
            .id("github-link")
            .text_sm()
            .cursor_pointer()
            .text_color(theme.link)
            .hover(|s| s.text_color(theme.link_hover))
            .on_click(cx.listener(|_, _, _, cx| {
                cx.open_url(REPO_URL);
            }))
            .child("View on GitHub →");

        v_flex()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .pt(TITLE_BAR_RESERVE)
            // Centered column, capped at the prose measure.
            .child(
                h_flex()
                    .flex_1()
                    .w_full()
                    .items_center()
                    .justify_center()
                    .child(
                        v_flex()
                            .w_full()
                            .max_w(rems(24.))
                            .px_8()
                            .gap_6()
                            .items_center()
                            .child(wordmark)
                            .child(version)
                            .child(purpose)
                            .child(license)
                            .child(github_link),
                    ),
            )
    }
}
