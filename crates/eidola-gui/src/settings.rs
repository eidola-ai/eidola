//! Settings window — a calm two-pane surface. A narrow nav list (General ·
//! Account · Wallet) sits on a `theme.sidebar` band down the left edge; the
//! selected pane renders in the content column. No primary-button tab strip,
//! no boxes-in-boxes: the nav is quiet text, the content is hairline rows.
//!
//! Settings deliberately keeps **no raw-data dumps** — measurement hex,
//! attestation documents, and the request log live in the Record window
//! (⇧⌘L); the panes here summarize and link there.

use gpui::{
    AppContext, Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement,
    Render, StatefulInteractiveElement, Styled, Window, div, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use crate::account::AccountView;
use crate::actions::CloseWindow;
use crate::general::GeneralView;
use crate::stores::Stores;
use crate::wallet::WalletView;

/// Vertical reserve at the top of the nav band so the macOS traffic lights
/// (at `point(14, 11)` per `lib.rs::transparent_titlebar`) sit on empty
/// sidebar rather than over the first nav item.
#[cfg(target_os = "macos")]
const NAV_TOP_RESERVE: gpui::Pixels = gpui::px(44.);
#[cfg(not(target_os = "macos"))]
const NAV_TOP_RESERVE: gpui::Pixels = gpui::px(12.);

/// Width of the nav band. Narrow on purpose — three words, not a sidebar.
const NAV_WIDTH: gpui::Pixels = gpui::px(132.);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SettingsPane {
    General,
    Account,
    Wallet,
}

impl SettingsPane {
    fn label(self) -> &'static str {
        match self {
            SettingsPane::General => "General",
            SettingsPane::Account => "Account",
            SettingsPane::Wallet => "Wallet",
        }
    }
}

pub struct SettingsView {
    selected: SettingsPane,
    general: Entity<GeneralView>,
    account: Entity<AccountView>,
    wallet: Entity<WalletView>,
    /// Focus handle the root tracks. We attach `CloseWindow`'s listener to
    /// the root; the focused node has to be at-or-below it for the listener
    /// to be in the dispatch path, so we `focus()` the handle on
    /// construction.
    focus_handle: FocusHandle,
}

impl SettingsView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let general = cx.new(|cx| GeneralView::new(stores.config.clone(), window, cx));
        let account = cx.new(|cx| AccountView::new(stores.clone(), window, cx));
        let wallet = cx.new(|cx| WalletView::new(stores, window, cx));

        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        Self {
            selected: SettingsPane::General,
            general,
            account,
            wallet,
            focus_handle,
        }
    }

    /// The focus handle the view tracks. Exposed so behavior tests can
    /// dispatch actions through it the same way real keystrokes would.
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    pub fn selected(&self) -> SettingsPane {
        self.selected
    }

    /// Switch panes. Public so the nav rows and behavior tests share one
    /// path.
    pub fn select(&mut self, pane: SettingsPane, cx: &mut Context<Self>) {
        if self.selected != pane {
            self.selected = pane;
            cx.notify();
        }
    }

    /// The General pane entity — exposed for behavior tests asserting the
    /// option-reveal state.
    pub fn general(&self) -> Entity<GeneralView> {
        self.general.clone()
    }

    /// The Account pane entity — exposed for behavior tests asserting the
    /// reset-confirm flow.
    pub fn account(&self) -> Entity<AccountView> {
        self.account.clone()
    }

    fn nav_item(&self, pane: SettingsPane, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let active = self.selected == pane;
        let mut item = div()
            .id(pane.label())
            .w_full()
            .px_2p5()
            .py_1()
            .rounded(px(6.))
            .cursor_pointer()
            .on_click(cx.listener(move |this, _, _, cx| this.select(pane, cx)))
            .child(pane.label());
        if active {
            item = item
                .bg(theme.sidebar_accent)
                .text_color(theme.sidebar_foreground);
        } else {
            item = item
                .text_color(theme.muted_foreground)
                .hover(|s| s.text_color(theme.sidebar_foreground));
        }
        item
    }
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        let body: gpui::AnyElement = match self.selected {
            SettingsPane::General => self.general.clone().into_any_element(),
            SettingsPane::Account => self.account.clone().into_any_element(),
            SettingsPane::Wallet => self.wallet.clone().into_any_element(),
        };

        h_flex()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .size_full()
            .items_start()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(
                v_flex()
                    .w(NAV_WIDTH)
                    .h_full()
                    .flex_none()
                    .bg(theme.sidebar)
                    .border_r_1()
                    .border_color(theme.sidebar_border)
                    .pt(NAV_TOP_RESERVE)
                    .px_2()
                    .gap_0p5()
                    .child(self.nav_item(SettingsPane::General, cx))
                    .child(self.nav_item(SettingsPane::Account, cx))
                    .child(self.nav_item(SettingsPane::Wallet, cx)),
            )
            // The scroll container needs the same width discipline as the
            // chat transcript (see the scroll-container invariant in
            // crates/eidola-gui/AGENTS.md): wrap it in a flex column that
            // owns the leftover width, and give the scroll div `.w_full()`
            // so taffy stretches it instead of content-sizing it — without
            // this, long pane text refuses to wrap and rows shrink to
            // content width.
            .child(
                v_flex().flex_1().h_full().min_w_0().child(
                    div()
                        .id("settings-body")
                        .w_full()
                        .flex_1()
                        .overflow_y_scroll()
                        .child(body),
                ),
            )
    }
}
