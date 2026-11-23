//! Settings, phone-style: one list off the Library, each row a door to a
//! screen of its own — the account, adding credit, the wallet, appearance,
//! updates, and what is running (About, with the licences).

use gpui::{
    Context, EventEmitter, InteractiveElement as _, IntoElement, ParentElement as _, Render,
    ScrollHandle, SharedString, StatefulInteractiveElement as _, Styled as _, Window, div,
};
use gpui_component::ActiveTheme as _;

use super::i18n::msg;
use super::nav::{Dest, NavEvent};
use super::stores::Stores;
use super::widgets::{OnClick, header, nav_row, section};

pub struct SettingsView {
    stores: Stores,
    scroll: ScrollHandle,
}

impl EventEmitter<NavEvent> for SettingsView {}

impl SettingsView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.account, |_, _, cx| cx.notify()).detach();
        cx.observe(&stores.updates, |_, _, cx| cx.notify()).detach();
        stores.updates.update(cx, |u, cx| u.refresh_report(cx));
        Self {
            stores,
            scroll: ScrollHandle::new(),
        }
    }
}

fn go(dest: Dest, cx: &mut Context<SettingsView>) -> OnClick {
    Box::new(cx.listener(move |_, _, _, cx| cx.emit(NavEvent::To(dest))))
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let account = self.stores.account.read(cx);
        let account_detail: SharedString = match account.config().account_id.as_deref() {
            Some(id) if account.has_account() => msg::settings_account_linked(cx, short_id(id)),
            _ => msg::settings_account_none(cx),
        };
        let has_account = account.has_account();
        let updates = self.stores.updates.read(cx);
        let failed = updates
            .report()
            .and_then(|r| r.as_ref().ok())
            .and_then(|r| r.failed.clone());
        let (updates_detail, updates_color) = match failed {
            Some(version) => (
                msg::settings_updates_failed(cx, version),
                Some(cx.theme().danger),
            ),
            None => (
                msg::settings_updates_detail(cx, env!("CARGO_PKG_VERSION")),
                None,
            ),
        };

        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "settings",
                msg::settings_title(cx),
                Some(back),
                None,
                cx,
            ))
            .child(
                div()
                    .id("settings-body")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .child(section(msg::settings_section_account(cx), cx))
                    .child(nav_row(
                        "settings/account",
                        msg::settings_account(cx),
                        Some(account_detail),
                        None,
                        go(Dest::Account, cx),
                        cx,
                    ))
                    .children(has_account.then(|| {
                        nav_row(
                            "settings/purchase",
                            msg::settings_purchase(cx),
                            Some(msg::settings_purchase_detail(cx)),
                            None,
                            go(Dest::Purchase, cx),
                            cx,
                        )
                    }))
                    .child(nav_row(
                        "settings/wallet",
                        msg::settings_wallet(cx),
                        Some(msg::settings_wallet_detail(cx)),
                        None,
                        go(Dest::Wallet, cx),
                        cx,
                    ))
                    .child(section(msg::settings_section_app(cx), cx))
                    .child(nav_row(
                        "settings/appearance",
                        msg::settings_appearance(cx),
                        Some(msg::settings_appearance_detail(cx)),
                        None,
                        go(Dest::Appearance, cx),
                        cx,
                    ))
                    .child(nav_row(
                        "settings/updates",
                        msg::settings_updates(cx),
                        Some(updates_detail),
                        updates_color,
                        go(Dest::Updates, cx),
                        cx,
                    ))
                    .child(nav_row(
                        "settings/about",
                        msg::settings_about(cx),
                        Some(msg::settings_about_detail(cx)),
                        None,
                        go(Dest::About, cx),
                        cx,
                    )),
            )
    }
}

/// The first group of an account id: enough to tell two apart, not the id.
pub fn short_id(id: &str) -> String {
    let head: String = id.chars().take(8).collect();
    if head.len() < id.len() {
        format!("{head}…")
    } else {
        head
    }
}
