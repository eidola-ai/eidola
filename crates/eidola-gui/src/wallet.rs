//! Wallet settings pane — the anonymous spend credentials, with their honest
//! lifecycle states.
//!
//! The listing comes from the local `credential_lifecycle` view (via
//! `Core::fetch_wallet` → `AppCore::wallet_lifecycle`), so every credential
//! the wallet has ever held is shown with the state the database actually
//! computes: **active** (spendable), **in flight** (a spend is unsettled —
//! recoverable), **spent** (settled into a successor), **expired** (issuer
//! key lapsed). No active/in-flight split into separate lists; one history,
//! newest first, hairline rules between rows.

use eidola_app_core::CredentialLifecycleInfo;
use gpui::{
    Context, Entity, IntoElement, ParentElement, Render, SharedString, Styled, Subscription,
    Window, div,
};
use gpui_component::{
    ActiveTheme, Disableable, Sizable, StyledExt, WindowExt,
    button::{Button, ButtonVariants},
    h_flex,
    notification::Notification,
    v_flex,
};

use crate::plans::format_credits;
use crate::stores::WalletStore;

pub struct WalletView {
    wallet: Entity<WalletStore>,
    _subscriptions: Vec<Subscription>,
}

impl WalletView {
    pub fn new(
        stores: crate::stores::Stores,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let wallet = stores.wallet.clone();
        let _subscriptions = vec![cx.observe(&wallet, |_, _, cx| cx.notify())];
        wallet.update(cx, |s, cx| s.refresh(cx));
        Self {
            wallet,
            _subscriptions,
        }
    }

    fn recover_all(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let weak_window = window.window_handle();
        self.wallet.update(cx, |s, cx| {
            s.recover(cx, move |_, recovered, cx| {
                let message = if recovered.is_empty() {
                    "No credentials could be recovered.".to_string()
                } else {
                    format!("Recovered {} credential(s).", recovered.len())
                };
                let _ = weak_window.update(cx, |_, window, cx| {
                    window.push_notification(Notification::info(SharedString::from(message)), cx);
                });
            });
        });
    }

    fn render_row(
        &self,
        idx: usize,
        cred: &CredentialLifecycleInfo,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let settled = matches!(cred.state.as_str(), "spent" | "expired");
        let nonce_short: String = cred.nonce.chars().take(16).collect();

        let (state_line, state_color) = match cred.state.as_str() {
            "active" => (
                format!("active · generation {}", cred.generation),
                theme.muted_foreground,
            ),
            "spending" => (
                format!(
                    "in flight — {} credits held",
                    format_credits(cred.spend_amount.unwrap_or(0))
                ),
                theme.warning,
            ),
            "spent" => (
                format!(
                    "spent — {} credits charged",
                    format_credits(cred.spend_amount.unwrap_or(0))
                ),
                theme.muted_foreground,
            ),
            "expired" => ("expired".to_string(), theme.muted_foreground),
            other => (other.to_string(), theme.muted_foreground),
        };

        let mut nonce_el = div()
            .text_sm()
            .font_family("Menlo")
            .child(SharedString::from(format!("{nonce_short}…")));
        let mut credits_el = div().child(SharedString::from(format!(
            "{} credits",
            format_credits(cred.credits)
        )));
        if settled {
            nonce_el = nonce_el.text_color(theme.muted_foreground);
            credits_el = credits_el.text_color(theme.muted_foreground);
        }

        let mut row = h_flex()
            .w_full()
            .py_2()
            .gap_3()
            .items_center()
            .child(
                v_flex().flex_1().gap_0p5().child(nonce_el).child(
                    div()
                        .text_xs()
                        .text_color(state_color)
                        .child(SharedString::from(state_line)),
                ),
            )
            .child(credits_el);
        if idx > 0 {
            row = row.border_t_1().border_color(theme.border);
        }
        row
    }
}

impl Render for WalletView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let store = self.wallet.read(cx);
        let busy = store.is_loading();
        let error = store.lifecycle().error().map(|e| e.to_string());
        let rows = store.lifecycle_rows().to_vec();
        let any_spending = rows.iter().any(|r| r.state == "spending");

        let mut col = v_flex().px_6().py_5().gap_3().w_full();

        let mut header = h_flex().w_full().justify_between().items_center().child(
            div()
                .text_sm()
                .font_medium()
                .text_color(theme.muted_foreground)
                .child("Credentials"),
        );
        let mut actions = h_flex().gap_2();
        if any_spending {
            actions = actions.child(
                Button::new("recover-all")
                    .small()
                    .label("Recover in-flight")
                    .disabled(busy)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.recover_all(window, cx);
                    })),
            );
        }
        actions = actions.child(
            Button::new("refresh-credentials")
                .ghost()
                .small()
                .label("Refresh")
                .on_click(cx.listener(|this, _, _, cx| {
                    this.wallet.update(cx, |s, cx| s.refresh(cx));
                })),
        );
        header = header.child(actions);
        col = col.child(header);

        col = col.child(div().text_xs().text_color(theme.muted_foreground).child(
            "Anonymous spend credentials provision themselves from your balance — \
                     the server cannot link them back to your account.",
        ));

        if rows.is_empty() {
            col = col.child(
                div()
                    .py_8()
                    .w_full()
                    .text_color(theme.muted_foreground)
                    .child("No credentials yet — they appear when you start chatting."),
            );
        } else {
            let mut list = v_flex().w_full().pt_1();
            for (idx, cred) in rows.iter().enumerate() {
                list = list.child(self.render_row(idx, cred, cx));
            }
            col = col.child(list);
        }

        if let Some(err) = error {
            col = col.child(
                div()
                    .text_sm()
                    .text_color(theme.danger)
                    .child(SharedString::from(err)),
            );
        }

        col
    }
}
