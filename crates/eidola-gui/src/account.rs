//! Account settings pane — balance, pools, plans, and the reset door.
//!
//! Money stays boring and honest: the balance is one line, pools are
//! hairline rows with humanized expiries, and the plans list reuses the
//! onboarding presentation (`crate::plans`) so purchase looks the same
//! everywhere. Reset is destructive-ish (it forgets the local account keys),
//! so it sits behind a two-step inline confirm — no modal.

use eidola_app_core::error::AppError;
use gpui::{
    AsyncApp, Context, Entity, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, Subscription, WeakEntity, Window, div,
};
use gpui_component::{
    ActiveTheme, Sizable, StyledExt,
    button::{Button, ButtonVariants},
    h_flex,
    label::Label,
    v_flex,
};

use crate::plans::{self, format_credits};
use crate::stores::{AccountStore, ConfigStore};

pub struct AccountView {
    config: Entity<ConfigStore>,
    account: Entity<AccountStore>,
    /// Two-step reset: the first click arms this; the second actually
    /// resets. Any other interaction (cancel) disarms.
    confirm_reset: bool,
    /// Price id of an in-flight checkout-session request, if any.
    checkout_pending: Option<String>,
    checkout_error: Option<String>,
    /// View-owned checkout task (the awaitable `request_checkout` is awaited
    /// here, in the view's own slot — it dies with the window).
    checkout_task: Option<gpui::Task<()>>,
    _subscriptions: Vec<Subscription>,
}

impl AccountView {
    pub fn new(
        stores: crate::stores::Stores,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let config = stores.config.clone();
        let account = stores.account.clone();
        let _subscriptions = vec![
            cx.observe(&config, |_, _, cx| cx.notify()),
            cx.observe(&account, |_, _, cx| cx.notify()),
        ];

        // Initial data: prices always; balances only when an account exists.
        // Each refresh owns its own task slot, so there is no debounce to
        // dodge.
        let has_account = config
            .read(cx)
            .state()
            .map(|s| s.has_account)
            .unwrap_or(false);
        account.update(cx, |s, cx| {
            s.refresh_prices(cx);
            if has_account {
                s.refresh_balances(cx);
            }
        });

        Self {
            config,
            account,
            confirm_reset: false,
            checkout_pending: None,
            checkout_error: None,
            checkout_task: None,
            _subscriptions,
        }
    }

    // --- Reset flow (public so behavior tests drive the same path) -------

    pub fn reset_armed(&self) -> bool {
        self.confirm_reset
    }

    pub fn request_reset(&mut self, cx: &mut Context<Self>) {
        self.confirm_reset = true;
        cx.notify();
    }

    pub fn cancel_reset(&mut self, cx: &mut Context<Self>) {
        self.confirm_reset = false;
        cx.notify();
    }

    pub fn confirm_reset(&mut self, cx: &mut Context<Self>) {
        if !self.confirm_reset {
            return;
        }
        self.confirm_reset = false;
        self.account.update(cx, |s, cx| s.reset_account(cx));
        cx.notify();
    }

    // --- Checkout (same flow as the onboarding plans page) ---------------

    pub fn checkout_pending(&self) -> Option<&str> {
        self.checkout_pending.as_deref()
    }

    pub fn begin_checkout(&mut self, price_id: String, cx: &mut Context<Self>) {
        if self.checkout_pending.is_some() {
            return;
        }
        self.checkout_pending = Some(price_id.clone());
        self.checkout_error = None;
        cx.notify();

        let Some(rx) = self.account.read(cx).request_checkout(price_id) else {
            // Stub core: the in-flight marker above is the observable state.
            return;
        };
        // Own the await in the view's own slot — the checkout request dies
        // with this window (per the doctrine's `request_*` shape).
        self.checkout_task = Some(cx.spawn(
            async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
                let res = rx.await.unwrap_or_else(|_| {
                    Err(AppError::Internal {
                        message: "checkout task cancelled".into(),
                    })
                });
                let _ = this.update(cx, |this, cx| {
                    this.checkout_pending = None;
                    this.checkout_task = None;
                    match res {
                        Ok(url) => cx.open_url(&url),
                        Err(e) => this.checkout_error = Some(e.to_string()),
                    }
                    cx.notify();
                });
            },
        ));
    }
}

impl Render for AccountView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let has_account = self
            .config
            .read(cx)
            .state()
            .map(|s| s.has_account)
            .unwrap_or(false);
        let account = self.account.read(cx);
        let balances = account.balances().value().cloned();
        let prices = account.prices().value().cloned().unwrap_or_default();
        let busy = account.is_loading();
        let core_error = account
            .balances()
            .error()
            .or_else(|| account.prices().error())
            .map(|e| e.to_string());

        let mut col = v_flex().px_6().py_5().gap_4().w_full();

        // --- Account ----------------------------------------------------
        col = col.child(section_header("Account", cx));
        if has_account {
            let mut account_block =
                v_flex().gap_1().child(div().child(
                    "Anonymous account on this machine — the server holds only a random id.",
                ));

            if self.confirm_reset {
                account_block = account_block.child(
                    v_flex()
                        .pt_2()
                        .gap_2()
                        .child(div().text_sm().text_color(theme.danger).child(
                            "This forgets the account keys on this device. Remaining \
                                     balance becomes unreachable; the local record of what was \
                                     spent stays in the Record.",
                        ))
                        .child(
                            h_flex()
                                .gap_2()
                                .child(
                                    Button::new("confirm-reset")
                                        .danger()
                                        .small()
                                        .label("Reset account")
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.confirm_reset(cx)),
                                        ),
                                )
                                .child(
                                    Button::new("cancel-reset")
                                        .ghost()
                                        .small()
                                        .label("Keep account")
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.cancel_reset(cx)),
                                        ),
                                ),
                        ),
                );
            } else {
                account_block = account_block.child(
                    h_flex().pt_1().text_xs().child(
                        div()
                            .id("request-reset")
                            .cursor_pointer()
                            .text_color(theme.muted_foreground)
                            .hover(|s| s.text_color(theme.danger))
                            .child("Reset account…")
                            .on_click(cx.listener(|this, _, _, cx| this.request_reset(cx))),
                    ),
                );
            }
            col = col.child(account_block);
        } else {
            col = col
                .child(
                    div()
                        .text_color(theme.muted_foreground)
                        .child("No account yet — a new space (⌘N) walks you through it."),
                )
                .child(
                    h_flex().child(
                        Button::new("create-account")
                            .primary()
                            .small()
                            .label("Create account")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.account.update(cx, |s, cx| s.create_account(cx));
                            })),
                    ),
                );
        }

        // --- Balance ------------------------------------------------------
        if has_account {
            col = col.child(div().pt_2().child(section_header("Balance", cx)));
            if let Some(b) = balances.as_ref() {
                col = col.child(
                    h_flex()
                        .items_baseline()
                        .gap_2()
                        .child(
                            div()
                                .text_xl()
                                .child(SharedString::from(format_credits(b.available))),
                        )
                        .child(
                            div()
                                .text_color(theme.muted_foreground)
                                .child("credits available"),
                        ),
                );
                let now = eidola_app_core::now_ms();
                for (idx, pool) in b.pools.iter().enumerate() {
                    let mut line =
                        format!("{} — {} credits", pool.source, format_credits(pool.amount));
                    if let Some(exp) = pool.expires_at {
                        line = format!("{line} · {}", humanize_expiry(exp, now));
                    }
                    let mut row = h_flex()
                        .w_full()
                        .py_1p5()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(line));
                    if idx > 0 {
                        row = row.border_t_1().border_color(theme.border);
                    }
                    col = col.child(row);
                }
            } else {
                col = col.child(div().text_color(theme.muted_foreground).child(if busy {
                    "Loading…"
                } else {
                    "Not loaded."
                }));
            }
            col = col.child(
                h_flex().text_xs().child(
                    div()
                        .id("refresh-balances")
                        .cursor_pointer()
                        .text_color(theme.muted_foreground)
                        .hover(|s| s.text_color(theme.foreground))
                        .child("Refresh")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.account.update(cx, |s, cx| s.refresh_balances(cx));
                        })),
                ),
            );
        }

        // --- Plans ----------------------------------------------------------
        col = col.child(div().pt_2().child(section_header("Plans", cx)));
        if prices.is_empty() {
            col = col.child(div().text_color(theme.muted_foreground).child(if busy {
                "Loading plans…"
            } else {
                "No plans loaded."
            }));
        } else {
            let weak = cx.entity().downgrade();
            let on_select: plans::PlanSelectHandler =
                std::rc::Rc::new(move |price_id, _window, app| {
                    let _ = weak.update(app, |this, cx| this.begin_checkout(price_id, cx));
                });
            col = col
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child("Checkout opens in your browser; credit lands on this account."),
                )
                .child(plans::plan_rows(
                    &prices,
                    self.checkout_pending.as_deref(),
                    on_select,
                    cx,
                ));
        }
        if let Some(err) = self.checkout_error.as_deref() {
            col = col.child(
                div()
                    .text_sm()
                    .text_color(theme.danger)
                    .child(SharedString::from(err.to_string())),
            );
        }

        if let Some(err) = core_error.as_deref() {
            col = col.child(error_banner(err, cx));
        }

        col
    }
}

/// Humanize a future expiry timestamp: "expires today", "expires in 5d",
/// "expires in 3w", … Falls to "expired" for past timestamps. Coarse on
/// purpose — the pools list wants a sense of urgency, not a deadline clock.
fn humanize_expiry(expires_ms: i64, now_ms: i64) -> String {
    const MINUTE: i64 = 60_000;
    const HOUR: i64 = 60 * MINUTE;
    const DAY: i64 = 24 * HOUR;

    let delta = expires_ms - now_ms;
    if delta < 0 {
        "expired".to_string()
    } else if delta < DAY {
        "expires today".to_string()
    } else if delta < 2 * DAY {
        "expires tomorrow".to_string()
    } else if delta < 14 * DAY {
        format!("expires in {}d", delta / DAY)
    } else if delta < 60 * DAY {
        format!("expires in {}w", delta / (7 * DAY))
    } else if delta < 365 * DAY {
        format!("expires in {}mo", delta / (30 * DAY))
    } else {
        format!("expires in {}y", delta / (365 * DAY))
    }
}

fn section_header(label: &str, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_color(theme.muted_foreground)
        .text_sm()
        .font_medium()
        .child(SharedString::from(label.to_string()))
}

fn error_banner(message: &str, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .gap_2()
        .px_3()
        .py_2()
        .rounded_md()
        .bg(theme.danger.opacity(0.08))
        .text_color(theme.danger)
        .child(Label::new(SharedString::from(message.to_string())))
}

#[cfg(test)]
mod tests {
    use super::humanize_expiry;

    const DAY: i64 = 24 * 60 * 60 * 1000;

    #[test]
    fn humanize_expiry_buckets() {
        let now = 1_900_000_000_000;
        assert_eq!(humanize_expiry(now - 1, now), "expired");
        assert_eq!(humanize_expiry(now + DAY / 2, now), "expires today");
        assert_eq!(humanize_expiry(now + DAY + 1, now), "expires tomorrow");
        assert_eq!(humanize_expiry(now + 5 * DAY, now), "expires in 5d");
        assert_eq!(humanize_expiry(now + 21 * DAY, now), "expires in 3w");
        assert_eq!(humanize_expiry(now + 90 * DAY, now), "expires in 3mo");
        assert_eq!(humanize_expiry(now + 400 * DAY, now), "expires in 1y");
    }
}
