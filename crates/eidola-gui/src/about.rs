//! About window — a small singleton showing the wordmark, version, a quiet
//! line of purpose, source note, and a "View on GitHub" link.
//!
//! Same transparent-titlebar treatment as all other windows. Singleton so
//! repeated "About Eidola" invocations raise the existing window rather than
//! stacking new ones. ~360×420 px.

use gpui::{
    Context, FocusHandle, InteractiveElement, IntoElement, ParentElement, Render,
    StatefulInteractiveElement, Styled, Window, div, px, relative, rems,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use crate::actions::CloseWindow;
use crate::i18n::msg;
use crate::probe::Probe as _;

/// The version string baked in at compile time.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// GitHub repository URL shown as the "View on GitHub" link.
const REPO_URL: &str = "https://github.com/eidola-ai/eidola";

/// Vertical reserve for the macOS traffic lights / Linux CSD window
/// controls (same pattern as all other windows with `transparent_titlebar`).
const TITLE_BAR_RESERVE: gpui::Pixels = crate::titlebar::DRAG_BAND_HEIGHT;

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
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        // Every visible string here comes from `locales/` through the generated
        // accessors (`crate::i18n`) — including the accessible labels, which
        // *are* the localized strings. Probe **names** never move: they are the
        // driver's stable selectors, and localizing them would break QA the way
        // localizing an id breaks a lookup.
        let version = msg::about_version_value(cx, VERSION);

        // Wordmark block: large "Eidola" + a hairline rule underneath,
        // matching the welcome page's title-page treatment.
        let wordmark = v_flex()
            .gap_3()
            .items_center()
            .child(
                div()
                    .id("about-title")
                    .probe("about/title", gpui::Role::Heading, msg::about_title(cx))
                    .aria_level(1)
                    .text_size(px(32.))
                    .line_height(relative(1.2))
                    .child(msg::about_title(cx)),
            )
            .child(div().w(rems(3.)).h(px(1.)).bg(theme.border));

        // Version line: muted, italic, small — unobtrusive. Unobtrusive is not
        // absent, though: it carries the version as its accessible value, so
        // "which build am I running" is answerable without sighted reading.
        let version_line = div()
            .id("about-version")
            .probe_value(
                "about/version",
                gpui::Role::Label,
                msg::about_version_label(cx),
                version.clone(),
            )
            .text_sm()
            .italic()
            .text_color(theme.muted_foreground)
            .child(version);

        // Purpose copy: echoes the welcome page's voice (same three-sentence
        // set minus the call to action — the reader has already begun).
        let purpose = v_flex()
            .gap_3()
            .text_sm()
            .text_color(theme.muted_foreground)
            .child(msg::about_purpose_lead(cx))
            .child(msg::about_purpose_attestation(cx));

        // Source note. Deliberately no license claim: the repository does
        // not yet carry a LICENSE file, and the About page must not assert
        // terms that aren't durably true (the no-fake-states rule applies
        // to legal claims too). Add the real license line when one lands.
        let license = div()
            .text_xs()
            .text_color(theme.muted_foreground)
            .child(msg::about_source_note(cx));

        // "View on GitHub" link — `cx.open_url` opens the default browser. The
        // accessible name is the bare verb; the visible text adds the arrow that
        // marks the link as leaving the app.
        let github_link = div()
            .id("github-link")
            .probe("about/github", gpui::Role::Link, msg::about_github(cx))
            .text_sm()
            .cursor_pointer()
            .text_color(theme.link)
            .hover(|s| s.text_color(theme.link_hover))
            .on_click(cx.listener(|_, _, _, cx| {
                cx.open_url(REPO_URL);
            }))
            .child(msg::about_github_cta(cx));

        crate::chrome::round_client_corners(v_flex(), window)
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .relative()
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
                            .child(version_line)
                            .child(purpose)
                            .child(license)
                            .child(github_link),
                    ),
            )
            // The drag band is the **last** child: a blocking hitbox only
            // suppresses hitboxes registered before it (see `crate::overlay`).
            .child(crate::titlebar::drag_band(
                "about-titlebar",
                TITLE_BAR_RESERVE,
                window,
                cx,
            ))
    }
}
