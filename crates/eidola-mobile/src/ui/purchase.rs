//! Adding credit: the plans, in the desktop's words (`plans.ftl`), and a
//! checkout that happens **in the system browser**, never in the app.
//!
//! - A tap mints a checkout link for that plan (a real request: the row says
//!   "Opening checkout…" until it lands) and hands it to the shell's
//!   `shell.openExternal`, which gives it to the operating system's browser.
//!   There is no payment in the app and no store purchase.
//! - **A link is opened only for the account it was minted for**, as the
//!   process that signed the request reports it (`CheckoutMint::minted_for`
//!   against the profile's credentials now): a mismatch is said, not opened.
//! - **With a subscription in force the recurring plans are not offered**
//!   (the server refuses a second); one-time top-ups stay. Only an answered,
//!   affirmative read narrows anything.
//! - Back from the browser, the balance is read again
//!   (`AccountStore::returned_to_foreground`).

use eidola_app_core::error::AppError;
use eidola_app_core::{CheckoutMint, PriceCadence, PriceInfo, SubscriptionState};
use gpui::{
    App, Context, EventEmitter, InteractiveElement as _, IntoElement, MouseButton,
    ParentElement as _, Render, Role, ScrollHandle, SharedString, StatefulInteractiveElement as _,
    Styled as _, Task, Window, div, prelude::FluentBuilder as _, px,
};
use gpui_component::ActiveTheme as _;

use super::GUTTER;
use super::account_view::format_credits;
use super::i18n::msg;
use super::loadable::Loadable;
use super::nav::NavEvent;
use super::probe::Probe as _;
use super::stores::Stores;
use super::widgets::{Pill, error_note, header, keep_keyboard, note, pill, section, value_row};

/// The plans a surface may offer: with a subscription in force, the
/// one-time top-ups only (the desktop's `plans::offered_plans`).
pub fn offered_plans(prices: &[PriceInfo], subscribed: bool) -> Vec<PriceInfo> {
    prices
        .iter()
        .filter(|p| !subscribed || p.cadence == PriceCadence::OneTime)
        .cloned()
        .collect()
}

#[derive(Debug)]
enum CheckoutFailure {
    StaleMint,
    Refused(AppError),
    Shell(String),
}

pub struct PurchaseView {
    stores: Stores,
    scroll: ScrollHandle,
    /// The plan whose checkout is being minted.
    pending: Option<String>,
    task: Option<Task<()>>,
    failure: Option<CheckoutFailure>,
    /// A checkout went to the browser in this visit.
    opened: bool,
}

impl EventEmitter<NavEvent> for PurchaseView {}

impl PurchaseView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.account, |_, _, cx| cx.notify()).detach();
        stores.account.update(cx, |a, cx| {
            a.refresh_prices(cx);
            a.refresh_subscription(cx);
            a.refresh_balances(cx);
        });
        Self {
            stores,
            scroll: ScrollHandle::new(),
            pending: None,
            task: None,
            failure: None,
            opened: false,
        }
    }

    fn checkout(&mut self, price_id: String, cx: &mut Context<Self>) {
        if self.pending.is_some() {
            return;
        }
        self.pending = Some(price_id.clone());
        self.failure = None;
        let mint = self.stores.account.read(cx).request_checkout(price_id);
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = mint.await;
            this.update(cx, |this, cx| this.land(result, cx)).ok();
        }));
        cx.notify();
    }

    fn land(&mut self, result: Result<CheckoutMint, AppError>, cx: &mut Context<Self>) {
        match result {
            Ok(mint)
                if self
                    .stores
                    .account
                    .read(cx)
                    .mint_is_current(&mint.minted_for) =>
            {
                let open = self.stores.host.open_external(mint.url);
                self.task = Some(cx.spawn(async move |this, cx| {
                    let opened = open.await;
                    this.update(cx, |this, cx| {
                        this.pending = None;
                        this.task = None;
                        match opened {
                            Ok(()) => {
                                this.opened = true;
                                this.stores.account.update(cx, |a, _| a.checkout_opened());
                            }
                            Err(e) => this.failure = Some(CheckoutFailure::Shell(e)),
                        }
                        cx.notify();
                    })
                    .ok();
                }));
                return;
            }
            Ok(_) => self.failure = Some(CheckoutFailure::StaleMint),
            Err(e) => self.failure = Some(CheckoutFailure::Refused(e)),
        }
        self.pending = None;
        self.task = None;
        cx.notify();
    }
}

/// A price's own line: what it costs and how often, in the reader's words.
fn price_line(price: &PriceInfo, cx: &App) -> SharedString {
    if price.amount.is_none() {
        return msg::plans_free(cx);
    }
    let amount = price.amount_display.clone();
    match &price.cadence {
        PriceCadence::OneTime => amount.into(),
        PriceCadence::Every { interval, count } => {
            msg::plans_price_cadence(cx, *count, interval.clone(), amount)
        }
    }
}

/// The credits a plan carries and when they expire.
fn credits_line(price: &PriceInfo, cx: &App) -> SharedString {
    let credits = format_credits(price.credits);
    let line = if price.cadence == PriceCadence::OneTime {
        msg::plans_credits_one_time(cx, price.credits, credits)
    } else {
        msg::plans_credits_recurring(cx, price.credits, credits)
    };
    match price
        .product_description
        .as_deref()
        .filter(|d| !d.trim().is_empty())
    {
        Some(description) => {
            msg::plans_credits_described(cx, line.to_string(), description.to_string())
        }
        None => line,
    }
}

impl Render for PurchaseView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let account = self.stores.account.read(cx);
        let prices = account.prices().clone();
        let subscribed = account
            .subscription()
            .value()
            .is_some_and(|s| s.state == SubscriptionState::Active);
        let has_account = account.has_account();
        let balance = account.balances().value().map(|b| b.available);
        let theme = cx.theme().clone();
        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));

        let mut body = div()
            .id("purchase-body")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.scroll)
            .pb(px(32.))
            .child(note(msg::purchase_intro(cx), cx))
            .child(note(msg::purchase_checkout_note(cx), cx));
        if !has_account {
            body = body.child(note(msg::account_none(cx), cx));
        } else {
            // The balance, read again when the reader comes back from the
            // browser.
            if let Some(available) = balance {
                body = body.child(value_row(
                    "purchase/balance",
                    msg::account_balance(cx),
                    msg::account_credits(cx, available, format_credits(available)),
                    false,
                    cx,
                ));
            }
            if subscribed {
                body = body.child(note(msg::purchase_subscribed_note(cx), cx));
            }
            body = body.child(section(msg::plans_list(cx), cx));
            match &prices {
                Loadable::NotLoaded | Loadable::Loading => {
                    body = body.child(note(msg::purchase_loading(cx), cx));
                }
                Loadable::Failed { prior: None, error } => {
                    body = body
                        .child(error_note(
                            "purchase/error",
                            msg::purchase_failed(cx, error.to_string()),
                            cx,
                        ))
                        .child(div().px(GUTTER).child(pill(
                            "purchase/retry",
                            msg::account_retry(cx),
                            Pill::Secondary,
                            true,
                            Box::new(cx.listener(|this, _, _, cx| {
                                this.stores.account.update(cx, |a, cx| a.refresh_prices(cx))
                            })),
                            cx,
                        )));
                }
                _ => {
                    let plans =
                        offered_plans(prices.value().map(Vec::as_slice).unwrap_or(&[]), subscribed);
                    if plans.is_empty() {
                        body = body.child(note(
                            if subscribed {
                                msg::purchase_none_topups(cx)
                            } else {
                                msg::purchase_none(cx)
                            },
                            cx,
                        ));
                    }
                    for (ix, price) in plans.iter().enumerate() {
                        let pending = self.pending.as_deref() == Some(price.id.as_str());
                        let busy = self.pending.is_some();
                        let line = if pending {
                            msg::plans_opening_checkout(cx)
                        } else {
                            price_line(price, cx)
                        };
                        let credits = credits_line(price, cx);
                        let id = price.id.clone();
                        let on_click =
                            cx.listener(move |this, _, _, cx| this.checkout(id.clone(), cx));
                        body = body.child(
                            div()
                                .id(("purchase-plan", ix))
                                .probe(
                                    format!("purchase/plan/{ix}"),
                                    Role::Button,
                                    format!("{}, {line}, {credits}", price.product_name),
                                )
                                .w_full()
                                .min_h(px(64.))
                                .px(GUTTER)
                                .py(px(10.))
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .border_b_1()
                                .border_color(theme.border)
                                .active(|s| s.bg(theme.secondary))
                                .on_mouse_down(MouseButton::Left, keep_keyboard)
                                .when(!busy, |d| d.on_click(on_click))
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
                                                .text_size(px(18.))
                                                .child(price.product_name.clone()),
                                        )
                                        .child(div().flex_none().text_size(px(15.)).child(line)),
                                )
                                .child(
                                    div()
                                        .text_size(px(14.))
                                        .text_color(theme.muted_foreground)
                                        .child(credits),
                                ),
                        );
                    }
                }
            }
            if let Some(failure) = &self.failure {
                let text = match failure {
                    CheckoutFailure::StaleMint => msg::account_stale_mint(cx),
                    CheckoutFailure::Refused(e) => msg::account_door_failed(cx, e.to_string()),
                    CheckoutFailure::Shell(e) => msg::account_door_failed(cx, e.clone()),
                };
                body = body.child(error_note("purchase/checkout-error", text, cx));
            }
            if self.opened {
                body = body.child(
                    div()
                        .id("purchase-opened")
                        .probe("purchase/opened", Role::Label, msg::purchase_opened(cx))
                        .child(note(msg::purchase_opened(cx), cx)),
                );
            }
        }
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "purchase",
                msg::purchase_title(cx),
                Some(back),
                None,
                cx,
            ))
            .child(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn price(id: &str, cadence: PriceCadence) -> PriceInfo {
        PriceInfo {
            id: id.into(),
            product_name: id.into(),
            product_description: None,
            amount_display: "$5.00".into(),
            recurrence: String::new(),
            credits: 5_000_000,
            amount: None,
            cadence,
        }
    }

    /// With a subscription in force only the one-time top-ups are offered
    /// (the server refuses a second subscription); otherwise every plan is.
    #[test]
    fn a_subscription_in_force_narrows_to_top_ups() {
        let prices = vec![
            price("once", PriceCadence::OneTime),
            price(
                "monthly",
                PriceCadence::Every {
                    interval: "month".into(),
                    count: 1,
                },
            ),
        ];
        let ids = |v: Vec<PriceInfo>| v.into_iter().map(|p| p.id).collect::<Vec<_>>();
        assert_eq!(ids(offered_plans(&prices, false)), ["once", "monthly"]);
        assert_eq!(ids(offered_plans(&prices, true)), ["once"]);
    }
}
