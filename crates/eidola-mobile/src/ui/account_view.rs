//! The account and the wallet: the desktop's Settings ▸ Account and Wallet
//! panes (`crates/eidola-gui/src/account.rs`, `wallet.rs`), on a phone.
//!
//! - **With no account, the screen offers the first run's account steps, not
//!   a creation of its own**: creating an account records acceptance of the
//!   terms, which may be submitted only for documents the reader was shown,
//!   and the consent step is the one screen that shows them.
//! - **The credentials are shown, the secret masked until asked for**, each
//!   with a Copy verb.
//! - **The balance and its pools**, read on every visit; the subscription's
//!   standing, with its billing door minted at the tap and opened only for
//!   the account it was minted for.
//! - **Reset is two steps**, inline: an armed warning, then "Reset account"
//!   or "Keep account".
//! - **State that describes an account does not outlive it**: the reveal,
//!   the armed reset and a billing door in flight are dropped the moment the
//!   credentials on display change, from wherever they changed.
//! - **The wallet is read-only**: every credential with where it stands
//!   (active, in flight, spent, expired).

use eidola_app_core::error::AppError;
use eidola_app_core::{CredentialLifecycleInfo, PortalMint, SubscriptionState};
use gpui::{
    App, Context, EventEmitter, InteractiveElement as _, IntoElement, ListAlignment, ListState,
    MouseButton, ParentElement as _, Render, Role, ScrollHandle, SharedString,
    StatefulInteractiveElement as _, Styled as _, Task, Window, div, list,
    prelude::FluentBuilder as _, px,
};
use gpui_component::{ActiveTheme as _, Icon};
use gpui_kit_assets::IconName;

use super::conversation::now_ms;
use super::i18n::msg;
use super::loadable::Loadable;
use super::nav::{Dest, NavEvent};
use super::probe::Probe as _;
use super::stores::Stores;
use super::widgets::{
    Pill, copy_row, error_note, header, keep_keyboard, note, pill, section, value_row,
};
use super::{GUTTER, TOUCH};

/// Credits grouped for reading ("1,250"), as the desktop app writes them.
pub fn format_credits(credits: i64) -> String {
    let raw = credits.unsigned_abs().to_string();
    let mut out = String::with_capacity(raw.len() + raw.len() / 3 + 1);
    if credits < 0 {
        out.push('-');
    }
    let offset = raw.len() % 3;
    for (i, ch) in raw.chars().enumerate() {
        if i > 0 && (i + 3 - offset).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// How far ahead a moment is, in the desktop's coarse buckets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ahead {
    Past,
    Today,
    Tomorrow,
    Days(i64),
    Weeks(i64),
    Months(i64),
    Years(i64),
}

pub fn ahead(then_ms: i64, now_ms: i64) -> Ahead {
    const DAY: i64 = 24 * 60 * 60_000;
    let delta = then_ms - now_ms;
    if delta < 0 {
        Ahead::Past
    } else if delta < DAY {
        Ahead::Today
    } else if delta < 2 * DAY {
        Ahead::Tomorrow
    } else if delta < 14 * DAY {
        Ahead::Days(delta / DAY)
    } else if delta < 60 * DAY {
        Ahead::Weeks(delta / (7 * DAY))
    } else if delta < 365 * DAY {
        Ahead::Months(delta / (30 * DAY))
    } else {
        Ahead::Years(delta / (365 * DAY))
    }
}

/// A pool's expiry, in words.
pub fn expiry_words(expires_ms: i64, cx: &App) -> SharedString {
    match ahead(expires_ms, now_ms()) {
        Ahead::Past => msg::account_expired(cx),
        Ahead::Today => msg::account_expires_today(cx),
        Ahead::Tomorrow => msg::account_expires_tomorrow(cx),
        Ahead::Days(n) => msg::account_expires_days(cx, n),
        Ahead::Weeks(n) => msg::account_expires_weeks(cx, n),
        Ahead::Months(n) => msg::account_expires_months(cx, n),
        Ahead::Years(n) => msg::account_expires_years(cx, n),
    }
}

/// Why a billing door was not opened.
#[derive(Debug)]
enum DoorFailure {
    /// The account changed while the link was minted: it was not opened.
    StaleMint,
    Refused(AppError),
    /// The shell could not hand the link to the browser.
    Shell(String),
}

pub struct AccountView {
    stores: Stores,
    scroll: ScrollHandle,
    /// The credentials this screen's state describes; when the profile's
    /// change, everything below is dropped.
    shown_for: Option<(String, String)>,
    reveal_secret: bool,
    reset_armed: bool,
    portal_task: Option<Task<()>>,
    portal_error: Option<DoorFailure>,
}

impl EventEmitter<NavEvent> for AccountView {}

impl AccountView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.account, |this, _, cx| {
            this.forget_if_identity_moved(cx);
            cx.notify();
        })
        .detach();
        // Live reads, nothing on the bus invalidates them: asked on every
        // visit.
        stores.account.update(cx, |a, cx| {
            a.refresh_balances(cx);
            a.refresh_subscription(cx);
        });
        let shown_for = identity(&stores, cx);
        Self {
            stores,
            scroll: ScrollHandle::new(),
            shown_for,
            reveal_secret: false,
            reset_armed: false,
            portal_task: None,
            portal_error: None,
        }
    }

    fn forget_if_identity_moved(&mut self, cx: &mut Context<Self>) {
        let now = identity(&self.stores, cx);
        if now != self.shown_for {
            self.shown_for = now;
            self.reveal_secret = false;
            self.reset_armed = false;
            self.portal_task = None;
            self.portal_error = None;
        }
    }

    fn open_portal(&mut self, cx: &mut Context<Self>) {
        if self.portal_task.is_some() {
            return;
        }
        self.portal_error = None;
        let mint = self.stores.account.read(cx).request_portal();
        self.portal_task = Some(cx.spawn(async move |this, cx| {
            let result = mint.await;
            this.update(cx, |this, cx| this.land_portal(result, cx))
                .ok();
        }));
        cx.notify();
    }

    fn land_portal(&mut self, result: Result<PortalMint, AppError>, cx: &mut Context<Self>) {
        self.portal_task = None;
        match result {
            Ok(mint)
                if self
                    .stores
                    .account
                    .read(cx)
                    .mint_is_current(&mint.minted_for) =>
            {
                let open = self.stores.host.open_external(mint.url);
                self.portal_task = Some(cx.spawn(async move |this, cx| {
                    let opened = open.await;
                    this.update(cx, |this, cx| {
                        this.portal_task = None;
                        if let Err(e) = opened {
                            this.portal_error = Some(DoorFailure::Shell(e));
                        } else {
                            this.stores.account.update(cx, |a, _| a.checkout_opened());
                        }
                        cx.notify();
                    })
                    .ok();
                }));
            }
            Ok(_) => self.portal_error = Some(DoorFailure::StaleMint),
            Err(e) => self.portal_error = Some(DoorFailure::Refused(e)),
        }
        cx.notify();
    }

    fn render_signed_out(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_3()
            .pt(px(16.))
            .child(note(msg::account_none(cx), cx))
            .child(div().px(GUTTER).child(pill(
                "account/set-up",
                msg::account_set_up(cx),
                Pill::Primary,
                true,
                Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::To(Dest::SetUpAccount)))),
                cx,
            )))
    }

    fn render_balance(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let balances = self.stores.account.read(cx).balances().clone();
        let mut col = div().flex().flex_col();
        match &balances {
            Loadable::NotLoaded | Loadable::Loading => {
                col = col.child(note(msg::account_balance_loading(cx), cx));
            }
            Loadable::Failed { prior: None, error } => {
                col = col
                    .child(error_note(
                        "account/balance/error",
                        msg::account_balance_failed(cx, error.to_string()),
                        cx,
                    ))
                    .child(div().px(GUTTER).child(pill(
                        "account/balance/retry",
                        msg::account_retry(cx),
                        Pill::Secondary,
                        true,
                        Box::new(cx.listener(|this, _, _, cx| {
                            this.stores
                                .account
                                .update(cx, |a, cx| a.refresh_balances(cx))
                        })),
                        cx,
                    )));
            }
            _ => {
                let Some(value) = balances.value() else {
                    return col;
                };
                let credits = format_credits(value.available);
                col = col.child(value_row(
                    "account/balance",
                    msg::account_balance(cx),
                    msg::account_credits(cx, value.available, credits),
                    false,
                    cx,
                ));
                for (idx, pool) in value.pools.iter().enumerate() {
                    let expiry = pool.expires_at.map(|ms| expiry_words(ms, cx));
                    let line = match expiry {
                        Some(when) => msg::account_pool_expiring(
                            cx,
                            pool.amount,
                            format_credits(pool.amount),
                            when.to_string(),
                        ),
                        None => msg::account_pool(cx, pool.amount, format_credits(pool.amount)),
                    };
                    col = col.child(
                        div()
                            .id(("account-pool", idx))
                            .probe(format!("account/pool/{idx}"), Role::Label, line.clone())
                            .px(GUTTER)
                            .py(px(2.))
                            .text_size(px(14.))
                            .text_color(cx.theme().muted_foreground)
                            .child(line),
                    );
                }
                if let Loadable::Failed { error, .. } = &balances {
                    col = col.child(error_note(
                        "account/balance/stale",
                        msg::account_balance_stale(cx, error.to_string()),
                        cx,
                    ));
                }
                col = col.child(div().px(GUTTER).pt(px(8.)).child(pill(
                    "account/balance/refresh",
                    msg::account_refresh(cx),
                    Pill::Secondary,
                    true,
                    Box::new(cx.listener(|this, _, _, cx| {
                        this.stores
                            .account
                            .update(cx, |a, cx| a.refresh_balances(cx))
                    })),
                    cx,
                )));
            }
        }
        col
    }

    fn render_subscription(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let subscription = self.stores.account.read(cx).subscription().clone();
        let opening = self.portal_task.is_some();
        let mut col = div().flex().flex_col().gap_2();
        let door = |name: &'static str, label: SharedString, cx: &mut Context<Self>| {
            div().px(GUTTER).child(pill(
                name,
                if opening {
                    msg::account_opening(cx)
                } else {
                    label
                },
                Pill::Secondary,
                !opening,
                Box::new(cx.listener(|this, _, _, cx| this.open_portal(cx))),
                cx,
            ))
        };
        match subscription.value() {
            None if subscription.error().is_some() => {
                col = col.child(error_note(
                    "account/subscription/error",
                    msg::account_subscription_failed(cx),
                    cx,
                ));
            }
            None => col = col.child(note(msg::account_subscription_checking(cx), cx)),
            Some(info) => {
                match info.state {
                    SubscriptionState::Active => {
                        let standing = match info.status.as_deref() {
                            Some("active") => msg::account_subscription_active(cx),
                            Some("trialing") => msg::account_subscription_trialing(cx),
                            Some("past_due") => msg::account_subscription_past_due(cx),
                            Some(other) => msg::account_subscription_other(cx, other.to_string()),
                            None => msg::account_subscription_in_force(cx),
                        };
                        col = col.child(note(standing, cx)).child(door(
                            "account/manage",
                            msg::account_manage(cx),
                            cx,
                        ));
                    }
                    SubscriptionState::Inactive => {
                        col = col
                            .child(note(msg::account_subscription_inactive(cx), cx))
                            .child(door("account/billing", msg::account_billing(cx), cx));
                    }
                    SubscriptionState::NoCustomer => {
                        col = col.child(note(msg::account_subscription_none(cx), cx));
                    }
                }
                if subscription.error().is_some() {
                    col = col.child(note(msg::account_subscription_stale(cx), cx));
                }
            }
        }
        if let Some(failure) = &self.portal_error {
            let text = match failure {
                DoorFailure::StaleMint => msg::account_stale_mint(cx),
                DoorFailure::Refused(e) => msg::account_door_failed(cx, e.to_string()),
                DoorFailure::Shell(e) => msg::account_door_failed(cx, e.clone()),
            };
            col = col.child(error_note("account/billing/error", text, cx));
        }
        col.child(div().px(GUTTER).child(pill(
            "account/subscription/check",
            msg::account_check_again(cx),
            Pill::Secondary,
            true,
            Box::new(cx.listener(|this, _, _, cx| {
                this.stores
                    .account
                    .update(cx, |a, cx| a.refresh_subscription(cx))
            })),
            cx,
        )))
    }

    fn render_reset(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut col = div().flex().flex_col().gap_2().pt(px(8.));
        if self.reset_armed {
            col = col
                .child(error_note(
                    "account/reset/warning",
                    msg::account_reset_warning(cx),
                    cx,
                ))
                .child(
                    div()
                        .px(GUTTER)
                        .flex()
                        .gap_2()
                        .child(pill(
                            "account/reset/confirm",
                            msg::account_reset_confirm(cx),
                            Pill::Danger,
                            true,
                            Box::new(cx.listener(|this, _, _, cx| {
                                this.reset_armed = false;
                                this.stores.account.update(cx, |a, cx| a.reset(cx));
                            })),
                            cx,
                        ))
                        .child(pill(
                            "account/reset/keep",
                            msg::account_reset_keep(cx),
                            Pill::Secondary,
                            true,
                            Box::new(cx.listener(|this, _, _, cx| {
                                this.reset_armed = false;
                                cx.notify();
                            })),
                            cx,
                        )),
                );
        } else {
            col = col.child(div().px(GUTTER).child(pill(
                "account/reset",
                msg::account_reset(cx),
                Pill::Secondary,
                true,
                Box::new(cx.listener(|this, _, _, cx| {
                    this.reset_armed = true;
                    cx.notify();
                })),
                cx,
            )));
        }
        col
    }
}

fn identity(stores: &Stores, cx: &App) -> Option<(String, String)> {
    let config = stores.account.read(cx).config();
    Some((config.account_id.clone()?, config.account_secret.clone()?))
}

impl Render for AccountView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let account = self.stores.account.read(cx);
        let credentials = account
            .has_account()
            .then(|| identity(&self.stores, cx))
            .flatten();
        let op_error = account.op_error().map(|e| e.to_string());
        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));
        let mut body = div()
            .id("account-body")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.scroll)
            .pb(px(32.));
        match credentials {
            None => body = body.child(self.render_signed_out(cx)),
            Some((id, secret)) => {
                let theme = cx.theme().clone();
                let reveal = self.reveal_secret;
                let shown_secret: SharedString = if reveal {
                    secret.clone().into()
                } else {
                    "•".repeat(24).into()
                };
                let reveal_label = if reveal {
                    msg::account_hide_secret(cx)
                } else {
                    msg::account_show_secret(cx)
                };
                body = body
                    .child(section(msg::account_section_credentials(cx), cx))
                    .child(copy_row("account/id", msg::account_id(cx), id.into(), cx))
                    .child(
                        div()
                            .flex()
                            .items_end()
                            .child(div().flex_1().min_w_0().child(copy_row(
                                "account/secret",
                                msg::account_secret(cx),
                                shown_secret,
                                cx,
                            )))
                            .child(
                                div()
                                    .id("account-secret-reveal")
                                    .probe("account/secret/reveal", Role::Button, reveal_label)
                                    .flex_none()
                                    .size(TOUCH)
                                    .mr(px(8.))
                                    .mb(px(8.))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_full()
                                    .on_mouse_down(MouseButton::Left, keep_keyboard)
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.reveal_secret = !this.reveal_secret;
                                        cx.notify();
                                    }))
                                    .child(
                                        Icon::new(if reveal {
                                            IconName::EyeOff
                                        } else {
                                            IconName::Eye
                                        })
                                        .size(px(20.))
                                        .text_color(theme.muted_foreground),
                                    ),
                            ),
                    )
                    .child(note(msg::account_secret_note(cx), cx))
                    .child(section(msg::account_section_balance(cx), cx))
                    .child(self.render_balance(cx))
                    .child(div().px(GUTTER).pt(px(12.)).child(pill(
                        "account/purchase",
                        msg::account_add_credit(cx),
                        Pill::Primary,
                        true,
                        Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::To(Dest::Purchase)))),
                        cx,
                    )))
                    .child(section(msg::account_section_subscription(cx), cx))
                    .child(self.render_subscription(cx))
                    .child(section(msg::account_section_reset(cx), cx))
                    .child(self.render_reset(cx));
            }
        }
        if let Some(error) = op_error {
            body = body.child(error_note(
                "account/op-error",
                msg::account_op_failed(cx, error),
                cx,
            ));
        }
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "account",
                msg::account_title(cx),
                Some(back),
                None,
                cx,
            ))
            .child(body)
    }
}

pub struct WalletView {
    stores: Stores,
    list: ListState,
    count: usize,
}

impl EventEmitter<NavEvent> for WalletView {}

impl WalletView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.wallet, |_, _, cx| cx.notify()).detach();
        stores.wallet.update(cx, |w, cx| w.refresh(cx));
        Self {
            stores,
            list: ListState::new(0, ListAlignment::Top, px(600.)),
            count: 0,
        }
    }

    fn render_row(&mut self, ix: usize, cx: &mut Context<Self>) -> gpui::AnyElement {
        if ix == 0 {
            return note(msg::wallet_note(cx), cx).into_any_element();
        }
        let Some(cred) = self
            .stores
            .wallet
            .read(cx)
            .lifecycle()
            .value()
            .and_then(|rows| rows.get(ix - 1))
            .cloned()
        else {
            return div().into_any_element();
        };
        wallet_row(ix - 1, &cred, cx).into_any_element()
    }
}

fn wallet_row(ix: usize, cred: &CredentialLifecycleInfo, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    let settled = matches!(cred.state.as_str(), "spent" | "expired");
    let held = cred.spend_amount.unwrap_or(0);
    let (state_line, color) = match cred.state.as_str() {
        "active" => (
            msg::wallet_active(cx, cred.generation),
            theme.muted_foreground,
        ),
        "spending" => (
            msg::wallet_spending(cx, held, format_credits(held)),
            theme.warning,
        ),
        "spent" => (
            msg::wallet_spent(cx, held, format_credits(held)),
            theme.muted_foreground,
        ),
        "expired" => (msg::wallet_expired(cx), theme.muted_foreground),
        other => (
            SharedString::from(other.to_string()),
            theme.muted_foreground,
        ),
    };
    let nonce: String = cred.nonce.chars().take(16).collect();
    let credits = msg::account_credits(cx, cred.credits, format_credits(cred.credits));
    div()
        .id(("wallet-row", ix))
        .probe(
            format!("wallet/row/{ix}"),
            Role::Label,
            format!("{credits}, {state_line}"),
        )
        .px(GUTTER)
        .py(px(10.))
        .flex()
        .items_center()
        .gap_3()
        .border_b_1()
        .border_color(theme.border)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(px(2.))
                .child(
                    div()
                        .font_family(super::theme::MONO_FAMILY)
                        .text_size(px(14.))
                        .when(settled, |d| d.text_color(theme.muted_foreground))
                        .child(format!("{nonce}…")),
                )
                .child(div().text_size(px(13.)).text_color(color).child(state_line)),
        )
        .child(
            div()
                .flex_none()
                .text_size(px(15.))
                .when(settled, |d| d.text_color(theme.muted_foreground))
                .child(credits),
        )
}

impl Render for WalletView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let lifecycle = self.stores.wallet.read(cx).lifecycle().clone();
        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));
        let refresh = div()
            .id("wallet-refresh")
            .probe("wallet/refresh", Role::Button, msg::account_refresh(cx))
            .min_h(TOUCH)
            .px(px(12.))
            .flex()
            .items_center()
            .text_size(px(15.))
            .text_color(cx.theme().link)
            .on_mouse_down(MouseButton::Left, keep_keyboard)
            .on_click(
                cx.listener(|this, _, _, cx| this.stores.wallet.update(cx, |w, cx| w.refresh(cx))),
            )
            .child(msg::account_refresh(cx))
            .into_any_element();
        let body = match &lifecycle {
            Loadable::NotLoaded | Loadable::Loading => div()
                .flex_1()
                .child(note(msg::account_balance_loading(cx), cx))
                .into_any_element(),
            Loadable::Failed { prior: None, error } => div()
                .flex_1()
                .child(error_note(
                    "wallet/error",
                    msg::wallet_failed(cx, error.to_string()),
                    cx,
                ))
                .into_any_element(),
            _ => {
                let rows = lifecycle.value().map(Vec::len).unwrap_or(0);
                if rows == 0 {
                    div()
                        .flex_1()
                        .child(note(msg::wallet_note(cx), cx))
                        .child(note(msg::wallet_empty(cx), cx))
                        .into_any_element()
                } else {
                    if rows + 1 != self.count {
                        self.count = rows + 1;
                        self.list.reset(self.count);
                    }
                    list(
                        self.list.clone(),
                        cx.processor(|this, ix, _, cx| this.render_row(ix, cx)),
                    )
                    .flex_1()
                    .size_full()
                    .into_any_element()
                }
            }
        };
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "wallet",
                msg::wallet_title(cx),
                Some(back),
                Some(refresh),
                cx,
            ))
            .child(div().flex_1().min_h_0().flex().flex_col().child(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credits_group_as_on_the_desktop() {
        assert_eq!(format_credits(0), "0");
        assert_eq!(format_credits(999), "999");
        assert_eq!(format_credits(1_000), "1,000");
        assert_eq!(format_credits(5_000_000), "5,000,000");
        assert_eq!(format_credits(-12_345), "-12,345");
    }

    #[test]
    fn expiry_buckets_as_on_the_desktop() {
        const DAY: i64 = 24 * 60 * 60_000;
        let now = 1_900_000_000_000;
        assert_eq!(ahead(now - 1, now), Ahead::Past);
        assert_eq!(ahead(now + DAY / 2, now), Ahead::Today);
        assert_eq!(ahead(now + DAY + 1, now), Ahead::Tomorrow);
        assert_eq!(ahead(now + 5 * DAY, now), Ahead::Days(5));
        assert_eq!(ahead(now + 21 * DAY, now), Ahead::Weeks(3));
        assert_eq!(ahead(now + 90 * DAY, now), Ahead::Months(3));
        assert_eq!(ahead(now + 400 * DAY, now), Ahead::Years(1));
    }
}
