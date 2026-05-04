use gpui::{
    AppContext, Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement,
    Render, StatefulInteractiveElement, Styled, Window, div,
};
use gpui_component::{
    ActiveTheme,
    button::{Button, ButtonVariants},
    h_flex, v_flex,
};

use crate::account::AccountView;
use crate::actions::CloseWindow;
use crate::core::Core;
use crate::general::GeneralView;
use crate::wallet::WalletView;

/// Left padding for the tab strip on macOS, large enough to clear the traffic
/// lights that sit at `point(14, 14)` and span ~68px wide. The tab row doubles
/// as the window's title bar — the lights live to the left of the tabs on a
/// shared band of `theme.background`. Matches gpui-component's own
/// `TITLE_BAR_LEFT_PADDING` (80px) for consistency with the platform norm.
#[cfg(target_os = "macos")]
const TAB_STRIP_LEFT_PAD: gpui::Pixels = gpui::px(80.);
#[cfg(not(target_os = "macos"))]
const TAB_STRIP_LEFT_PAD: gpui::Pixels = gpui::px(12.);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    General,
    Account,
    Wallet,
}

pub struct SettingsView {
    selected: Tab,
    general: Entity<GeneralView>,
    account: Entity<AccountView>,
    wallet: Entity<WalletView>,
    /// Focus handle the root v_flex tracks. We attach `CloseWindow`'s
    /// listener to the v_flex; the focused node has to be at-or-below
    /// that v_flex for the listener to be in the dispatch path, so we
    /// `focus()` the handle on construction.
    focus_handle: FocusHandle,
}

impl SettingsView {
    pub fn new(core: Entity<Core>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let general = cx.new(|cx| GeneralView::new(core.clone(), window, cx));
        let account = cx.new(|cx| AccountView::new(core.clone(), window, cx));
        let wallet = cx.new(|cx| WalletView::new(core, window, cx));

        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        Self {
            selected: Tab::General,
            general,
            account,
            wallet,
            focus_handle,
        }
    }

    fn tab_button(
        &self,
        id: &'static str,
        label: &'static str,
        tab: Tab,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected = self.selected == tab;
        let mut button =
            Button::new(id)
                .label(label)
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.selected = tab;
                    cx.notify();
                }));
        if selected {
            button = button.primary();
        } else {
            button = button.ghost();
        }
        button
    }
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        let body: gpui::AnyElement = match self.selected {
            Tab::General => self.general.clone().into_any_element(),
            Tab::Account => self.account.clone().into_any_element(),
            Tab::Wallet => self.wallet.clone().into_any_element(),
        };

        v_flex()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .pl(TAB_STRIP_LEFT_PAD)
                    .pr_3()
                    .py_2()
                    .border_b_1()
                    .border_color(theme.border)
                    .child(self.tab_button("tab-general", "General", Tab::General, cx))
                    .child(self.tab_button("tab-account", "Account", Tab::Account, cx))
                    .child(self.tab_button("tab-wallet", "Wallet", Tab::Wallet, cx)),
            )
            .child(
                div()
                    .id("settings-body")
                    .flex_1()
                    .overflow_y_scroll()
                    .child(body),
            )
    }
}
