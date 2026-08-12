//! Account settings pane — balance, pools, plans, and the reset door.
//!
//! Money stays boring and honest: the balance is one line, pools are
//! hairline rows with humanized expiries, and the plans list reuses the
//! onboarding presentation (`crate::plans`) so purchase looks the same
//! everywhere. Reset is destructive-ish (it forgets the local account keys),
//! so it sits behind a two-step inline confirm — no modal.

use eidola_app_core::SubscriptionState;
use eidola_app_core::error::AppError;
use gpui::{
    App, AppContext, AsyncApp, ClipboardItem, Context, Entity, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Subscription,
    WeakEntity, Window, div, prelude::FluentBuilder,
};
use gpui_component::{
    ActiveTheme, IconName, Sizable, StyledExt,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputEvent, InputState},
    label::Label,
    v_flex,
};

use crate::plans::{self, format_credits};
use crate::probe::Probe as _;
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
    /// "Manage subscription" mints a fresh billing-portal session on the
    /// click, so it is a real request with its own in-flight marker, its own
    /// error, and its own view-owned slot — the checkout shape exactly.
    manage_pending: bool,
    manage_error: Option<String>,
    manage_task: Option<gpui::Task<()>>,
    account_id_input: Entity<InputState>,
    account_secret_input: Entity<InputState>,
    account_id_seed: Option<SharedString>,
    account_secret_seed: Option<SharedString>,
    account_secret_revealed: bool,
    _subscriptions: Vec<Subscription>,
}

impl AccountView {
    pub fn new(stores: crate::stores::Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let config = stores.config.clone();
        let account = stores.account.clone();
        let account_id_input = cx.new(|cx| InputState::new(window, cx));
        let account_secret_input = cx.new(|cx| InputState::new(window, cx).masked(true));
        let _subscriptions = vec![
            cx.observe(&config, |_, _, cx| cx.notify()),
            cx.observe(&account, |_, _, cx| cx.notify()),
            cx.subscribe_in(
                &account_id_input,
                window,
                |this, _, ev: &InputEvent, window, cx| {
                    if matches!(ev, InputEvent::Change) {
                        this.sync_account_credential_inputs(window, cx);
                    }
                },
            ),
            cx.subscribe_in(
                &account_secret_input,
                window,
                |this, _, ev: &InputEvent, window, cx| {
                    if matches!(ev, InputEvent::Change) {
                        this.sync_account_credential_inputs(window, cx);
                    }
                },
            ),
        ];

        // Nothing is fetched here. `SettingsView` builds every pane when the
        // *window* opens, so a constructor-time read is a read for a reader
        // who may never come, and the only one they ever get if they do —
        // see `pane_activated`.
        Self {
            config,
            account,
            confirm_reset: false,
            checkout_pending: None,
            checkout_error: None,
            checkout_task: None,
            manage_pending: false,
            manage_error: None,
            manage_task: None,
            account_id_input,
            account_secret_input,
            account_id_seed: None,
            account_secret_seed: None,
            account_secret_revealed: false,
            _subscriptions,
        }
    }

    /// The reader has selected this pane — ask for everything it shows.
    ///
    /// This is where the pane's remote reads live, not the constructor:
    /// `SettingsView` eagerly builds all six panes at window creation, so
    /// construction happens once, for whoever opened Settings, whatever pane
    /// they then chose. Every one of these cells is a live server read that
    /// **no `Change` can invalidate** — the balance moves when a webhook
    /// credits the account and the subscription moves when the reader
    /// cancels in the browser portal, and neither of those is a local commit
    /// for the bus to announce. Selecting the pane is therefore the moment
    /// it is honest to ask.
    ///
    /// Each cell owns its own supersede slot, so re-selecting the pane
    /// quickly replaces an in-flight read rather than stacking on it, and a
    /// re-read over a value renders `Loaded { stale }` — never a blank.
    pub fn pane_activated(&mut self, cx: &mut Context<Self>) {
        let has_account = self
            .config
            .read(cx)
            .state()
            .map(|s| s.has_account)
            .unwrap_or(false);
        self.account.update(cx, |s, cx| {
            s.refresh_prices(cx);
            if has_account {
                s.refresh_balances(cx);
                s.refresh_subscription(cx);
            }
        });
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

    // --- Manage subscription (the billing portal) ------------------------

    pub fn manage_pending(&self) -> bool {
        self.manage_pending
    }

    /// Open the billing portal. The portal link is a short-lived session, so
    /// this asks for a fresh one now rather than opening the one the pane
    /// fetched when it opened.
    pub fn begin_manage(&mut self, cx: &mut Context<Self>) {
        if self.manage_pending {
            return;
        }
        self.manage_pending = true;
        self.manage_error = None;
        cx.notify();

        let Some(rx) = self.account.read(cx).request_portal() else {
            // Stub core: the in-flight marker above is the observable state.
            return;
        };
        self.manage_task = Some(cx.spawn(
            async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
                let res = rx.await.unwrap_or_else(|_| {
                    Err(AppError::Internal {
                        message: "billing portal task cancelled".into(),
                    })
                });
                let _ = this.update(cx, |this, cx| {
                    this.manage_pending = false;
                    this.manage_task = None;
                    match res {
                        Ok(url) => cx.open_url(&url),
                        // The server refuses a portal for an account with no
                        // payment relationship, which is not a state this
                        // door is offered in — so anything landing here is
                        // the mint itself failing, and the error says so.
                        Err(e) => this.manage_error = Some(e.to_string()),
                    }
                    cx.notify();
                });
            },
        ));
    }

    fn account_credentials(&self, cx: &App) -> (Option<SharedString>, Option<SharedString>) {
        let config = self.config.read(cx);
        let state = config.state();
        (
            state
                .and_then(|s| s.account_id.as_ref())
                .map(|s| SharedString::from(s.clone())),
            state
                .and_then(|s| s.account_secret.as_ref())
                .map(|s| SharedString::from(s.clone())),
        )
    }

    fn sync_account_credential_inputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (account_id, account_secret) = self.account_credentials(cx);
        sync_readonly_input(
            &self.account_id_input,
            &mut self.account_id_seed,
            account_id,
            window,
            cx,
        );
        sync_readonly_input(
            &self.account_secret_input,
            &mut self.account_secret_seed,
            account_secret,
            window,
            cx,
        );
        self.account_secret_input.update(cx, |s, cx| {
            s.set_masked(!self.account_secret_revealed, window, cx)
        });
    }

    fn toggle_account_secret_revealed(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.account_secret_revealed = !self.account_secret_revealed;
        self.account_secret_input.update(cx, |s, cx| {
            s.set_masked(!self.account_secret_revealed, window, cx)
        });
        cx.notify();
    }
}

impl Render for AccountView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync_account_credential_inputs(window, cx);
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
        let subscription = account.subscription().value().cloned();
        let subscription_error = account.subscription().error().map(|e| e.to_string());
        let subscription_loading = account.subscription().is_loading();
        // Only an answered *and* affirmative read filters the plans. An
        // unanswered or failed one leaves every plan offered: the server is
        // the authority on whether a second subscription is allowed, and it
        // refuses honestly, whereas hiding plans on a guess would strand a
        // reader who has none.
        let subscribed = subscription
            .as_ref()
            .is_some_and(|s| s.state == SubscriptionState::Active);
        let core_error = account
            .balances()
            .error()
            .or_else(|| account.prices().error())
            .map(|e| e.to_string());
        // The last account create/reset failure, if any — surfaced so the
        // Settings button never silently does nothing (honest-states rule).
        let account_op_error = account.account_op_error().map(|e| e.to_string());

        let mut col = v_flex().px_6().py_5().gap_4().w_full();

        // --- Account ----------------------------------------------------
        col = col.child(section_header("Account", cx));
        if has_account {
            let (account_id, account_secret) = self.account_credentials(cx);
            let mut account_block = v_flex().gap_3();
            if let Some(account_id) = account_id {
                account_block = account_block.child(account_credential_input(
                    "Account ID",
                    "settings/account/id",
                    &self.account_id_input,
                    account_id,
                    false,
                    self.account_secret_revealed,
                    |_, _, _| {},
                    cx,
                ));
            }
            if let Some(account_secret) = account_secret {
                account_block = account_block.child(account_credential_input(
                    "Account Secret",
                    "settings/account/secret",
                    &self.account_secret_input,
                    account_secret,
                    true,
                    self.account_secret_revealed,
                    cx.listener(|this, _, window, cx| {
                        this.toggle_account_secret_revealed(window, cx)
                    }),
                    cx,
                ));
            }

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
                                    // Probed wrapper for the a11y role/label —
                                    // shrink-wraps the button so its bounds are
                                    // an honest click target.
                                    div()
                                        .id("confirm-reset-wrap")
                                        .probe(
                                            "settings/account/reset-confirm",
                                            gpui::Role::Button,
                                            "Reset account",
                                        )
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.confirm_reset(cx)),
                                        )
                                        .child(
                                            Button::new("confirm-reset")
                                                .role(None)
                                                .danger()
                                                .small()
                                                .label("Reset account")
                                                .tab_stop(false),
                                        ),
                                )
                                .child(
                                    div()
                                        .id("cancel-reset-wrap")
                                        .probe(
                                            "settings/account/reset-cancel",
                                            gpui::Role::Button,
                                            "Keep account",
                                        )
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.cancel_reset(cx)),
                                        )
                                        .child(
                                            Button::new("cancel-reset")
                                                .role(None)
                                                .ghost()
                                                .small()
                                                .label("Keep account")
                                                .tab_stop(false),
                                        ),
                                ),
                        ),
                );
            } else {
                account_block = account_block.child(
                    h_flex().pt_1().text_xs().child(
                        div()
                            .id("request-reset")
                            .probe(
                                "settings/account/reset",
                                gpui::Role::Button,
                                "Reset account…",
                            )
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
                .child(div().text_color(theme.muted_foreground).child(format!(
                    "No account yet — a new space ({}) walks you through it.",
                    crate::actions::primary_chord("N")
                )))
                .child(
                    h_flex().child(
                        // Probed wrapper for the a11y role/label — shrink-wraps
                        // the button so its bounds are an honest click target.
                        div()
                            .id("create-account-wrap")
                            .probe(
                                "settings/account/create",
                                gpui::Role::Button,
                                "Create account",
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.account.update(cx, |s, cx| s.create_account(cx));
                            }))
                            .child(
                                Button::new("create-account")
                                    .role(None)
                                    .primary()
                                    .small()
                                    .label("Create account")
                                    .tab_stop(false),
                            ),
                    ),
                );
        }

        // Account create/reset failure — rendered right under the Account
        // controls so a failed button click is never silent.
        if let Some(err) = account_op_error.as_deref() {
            col = col.child(
                div()
                    .id("account-error")
                    .probe("settings/account/error", gpui::Role::Alert, err.to_string())
                    .child(error_banner(err, cx)),
            );
        }

        // --- Balance ------------------------------------------------------
        if has_account {
            col = col.child(div().pt_2().child(section_header("Balance", cx)));
            if let Some(b) = balances.as_ref() {
                col = col.child(
                    h_flex()
                        .id("account-balance")
                        // The figure and its unit are two node-less `div`s, so
                        // the balance reached assistive technology nowhere. It
                        // rides as the value under a stable name, which is also
                        // what keeps a refreshed figure from renaming the node.
                        .probe_value(
                            "settings/account/balance",
                            gpui::Role::Label,
                            "Balance",
                            SharedString::from(format!(
                                "{} credits available",
                                format_credits(b.available)
                            )),
                        )
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
                        .id(("account-pool", idx))
                        // Source, amount and humanized expiry are one rendered
                        // line; it is the value, indexed so repeated pools are
                        // distinguishable by name (S8's repeated-label rule).
                        .probe_value(
                            format!("settings/account/pool/{idx}"),
                            gpui::Role::Label,
                            format!("Credit pool {}", idx + 1),
                            SharedString::from(line.clone()),
                        )
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
                        .probe(
                            "settings/account/refresh-balances",
                            gpui::Role::Button,
                            "Refresh balances",
                        )
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

        // --- Subscription ---------------------------------------------------
        // Only with an account: without one there is nothing to have a
        // subscription, and the pane never asked.
        if has_account {
            // A failed read with nothing behind it is a blank the reader
            // can't get past — the panel says so and offers the way back.
            // A failed *re-read* over a known answer is not: that keeps its
            // value on screen (below) with a quiet note beside it, so the
            // section never blanks over a refresh error.
            let failed_cold = subscription_error.is_some() && subscription.is_none();
            // Any answered read draws the section — including `NoCustomer`.
            // What that state withholds is the **billing door**, not the
            // answer: an account money has never moved for has no payment
            // relationship to be let into, and a portal session minted for
            // it would be a door onto an empty room. It still needs the
            // answer and the way to ask again, because the reader who
            // completes their first checkout is in exactly this state with
            // the pane already open in front of them.
            let answered = subscription.is_some();

            if failed_cold {
                let err = subscription_error.as_deref().unwrap_or_default();
                col = col.child(div().pt_2().child(section_header("Subscription", cx)));
                col = col.child(crate::participants::load_error_panel(
                    "settings/account/subscription-retry",
                    "Couldn't check your subscription",
                    err,
                    cx,
                    cx.listener(|this, _, _, cx| {
                        this.account.update(cx, |s, cx| s.refresh_subscription(cx));
                    }),
                ));
            } else if subscription_loading {
                col = col
                    .child(div().pt_2().child(section_header("Subscription", cx)))
                    .child(
                        div()
                            .id("subscription-loading")
                            .probe_value(
                                "settings/account/subscription",
                                gpui::Role::Label,
                                "Subscription",
                                SharedString::from("Checking your subscription…"),
                            )
                            .text_color(theme.muted_foreground)
                            .child("Checking your subscription…"),
                    );
            } else if answered {
                let info = subscription.as_ref().expect("answered implies a value");
                // Three standings, three sentences — and only two doors. The
                // lapsed one is not offered "manage your subscription",
                // because there isn't one; what it still has is a payment
                // relationship, which the server now asserts only for an
                // account money has actually moved for. `NoCustomer` gets
                // the sentence and the re-check but no door at all.
                let (summary, door) = match info.state {
                    SubscriptionState::Active => (
                        subscription_summary(
                            info.status.as_deref(),
                            info.current_period_end,
                            eidola_app_core::now_ms(),
                        ),
                        Some((
                            "Manage subscription",
                            "settings/account/manage-subscription",
                            "Opens our payment processor's billing portal in your browser.",
                        )),
                    ),
                    SubscriptionState::Inactive => (
                        "You don't have a subscription right now.".to_string(),
                        Some((
                            "Billing and receipts",
                            "settings/account/billing-portal",
                            "Opens our payment processor's billing portal in your browser, \
                             where your payment methods, invoices and past receipts live.",
                        )),
                    ),
                    SubscriptionState::NoCustomer => (
                        "You don't have a subscription on this account.".to_string(),
                        None,
                    ),
                };

                col = col
                    .child(div().pt_2().child(section_header("Subscription", cx)))
                    .child(
                        div()
                            .id("subscription-summary")
                            .probe_value(
                                "settings/account/subscription",
                                gpui::Role::Label,
                                "Subscription",
                                SharedString::from(summary.clone()),
                            )
                            .child(SharedString::from(summary)),
                    );

                // The door is offered on the **standing**, and the standing is
                // all the read carries: minting a portal session is its own
                // request, made at the click. A door gated on a link fetched
                // when the pane opened would vanish for a paying customer
                // over a blip in a call they never needed.
                if let Some((portal_label, portal_probe, portal_note)) = door {
                    col = col
                        .child(
                            h_flex().pt_1().child(
                                // Probed wrapper for the a11y role/label —
                                // shrink-wraps the button so its bounds are an
                                // honest click target.
                                div()
                                    .id("billing-portal-wrap")
                                    .probe(portal_probe, gpui::Role::Button, portal_label)
                                    .on_click(cx.listener(|this, _, _, cx| this.begin_manage(cx)))
                                    .child({
                                        // Solid while a subscription stands —
                                        // managing it is the reason to be in
                                        // this section. Lapsed, the same
                                        // accent steps back to an outline: the
                                        // plans below are what that reader
                                        // most likely came for, and two solid
                                        // buttons on one pane are two shouts.
                                        let door =
                                            Button::new("billing-portal").role(None).primary();
                                        let door = if subscribed { door } else { door.outline() };
                                        door.small()
                                            .label(if self.manage_pending {
                                                "Opening…"
                                            } else {
                                                portal_label
                                            })
                                            .tab_stop(false)
                                    }),
                            ),
                        )
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.muted_foreground)
                                .child(portal_note),
                        );
                }

                if let Some(err) = self.manage_error.as_deref() {
                    col = col.child(
                        div()
                            .id("manage-subscription-error")
                            .probe(
                                "settings/account/manage-error",
                                gpui::Role::Alert,
                                err.to_string(),
                            )
                            .text_sm()
                            .text_color(theme.danger)
                            .child(SharedString::from(err.to_string())),
                    );
                }

                // A re-read that failed over a known answer keeps that answer
                // on screen — above, and in what the plans below offer — so
                // all it owes the reader is that it may be out of date.
                if subscription_error.is_some() {
                    col = col.child(
                        div()
                            .pt_1()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(
                                "Couldn't re-check your subscription — showing the last answer.",
                            ),
                    );
                }

                // **The way to ask again, always.** Nothing on the bus can
                // invalidate this cell, and the reader's own act of changing
                // it happens in a browser window this app never hears about:
                // they cancel in the portal, come back to a Settings window
                // that never closed, and the pane has no reason to have
                // asked since. Selecting the pane re-reads
                // (`pane_activated`), but a reader who never left it needs a
                // door, and one offered only after a failure is a door that
                // opens only when it is already too late.
                col = col.child(
                    h_flex().pt_1().text_xs().child(
                        div()
                            .id("subscription-recheck")
                            .probe(
                                "settings/account/subscription-retry",
                                gpui::Role::Button,
                                "Check again",
                            )
                            .cursor_pointer()
                            .text_color(theme.muted_foreground)
                            .hover(|s| s.text_color(theme.foreground))
                            .child("Check again")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.account.update(cx, |s, cx| s.refresh_subscription(cx));
                            })),
                    ),
                );
            }
        }

        // --- Plans ----------------------------------------------------------
        // A subscription in force means the server refuses a second one, so
        // the recurring plans come out and the one-time top-ups stay.
        let offered = plans::offered_plans(&prices, subscribed);
        if offered.is_empty() {
            col = col.child(div().text_color(theme.muted_foreground).child(if busy {
                "Loading plans…"
            } else if subscribed {
                "No one-time top-ups are available right now."
            } else {
                "No plans loaded."
            }));
        } else {
            let weak = cx.entity().downgrade();
            let on_select: plans::PlanSelectHandler =
                std::rc::Rc::new(move |price_id, _window, app| {
                    let _ = weak.update(app, |this, cx| this.begin_checkout(price_id, cx));
                });
            col = col.child(plans::plan_rows(
                &offered,
                self.checkout_pending.as_deref(),
                on_select,
                "settings/account",
                cx,
            ));
        }
        if let Some(err) = self.checkout_error.as_deref() {
            col = col.child(
                div()
                    .id("account-checkout-error")
                    .probe(
                        "settings/account/checkout-error",
                        gpui::Role::Alert,
                        err.to_string(),
                    )
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

/// Coarse relative phrasing for a *future* instant: "today", "tomorrow",
/// "in 5d", "in 3w", … `None` once the instant is past, because the
/// sentences that follow differ (credits *expire*, a billing period
/// *ends*) and only the caller knows which it is writing. Coarse on
/// purpose — these surfaces want a sense of when, not a deadline clock.
fn relative_future(then_ms: i64, now_ms: i64) -> Option<String> {
    const MINUTE: i64 = 60_000;
    const HOUR: i64 = 60 * MINUTE;
    const DAY: i64 = 24 * HOUR;

    let delta = then_ms - now_ms;
    Some(if delta < 0 {
        return None;
    } else if delta < DAY {
        "today".to_string()
    } else if delta < 2 * DAY {
        "tomorrow".to_string()
    } else if delta < 14 * DAY {
        format!("in {}d", delta / DAY)
    } else if delta < 60 * DAY {
        format!("in {}w", delta / (7 * DAY))
    } else if delta < 365 * DAY {
        format!("in {}mo", delta / (30 * DAY))
    } else {
        format!("in {}y", delta / (365 * DAY))
    })
}

/// Humanize a credit pool's expiry: "expires today", "expires in 5d", …
fn humanize_expiry(expires_ms: i64, now_ms: i64) -> String {
    match relative_future(expires_ms, now_ms) {
        Some(when) => format!("expires {when}"),
        None => "expired".to_string(),
    }
}

/// The one-line reading of an in-force subscription: what its status means
/// and, when known, when the current period ends. `status` is the payment
/// processor's own word for it; anything outside the three in-force
/// statuses is reported plainly rather than dressed up, because being
/// wrong about someone's billing is worse than being terse.
fn subscription_summary(status: Option<&str>, period_end_ms: Option<i64>, now_ms: i64) -> String {
    let standing = match status {
        Some("active") => "Your subscription is active.".to_string(),
        Some("trialing") => "Your subscription is in its trial period.".to_string(),
        Some("past_due") => {
            "Your subscription is active, but a payment did not go through. Update your \
             payment method to keep it."
                .to_string()
        }
        Some(other) => format!("Your subscription is in force, reported as “{other}”."),
        None => "Your subscription is in force.".to_string(),
    };
    let Some(end) = period_end_ms else {
        return standing;
    };
    match relative_future(end, now_ms) {
        Some(when) => format!("{standing} The current billing period ends {when}."),
        None => format!("{standing} The current billing period has ended."),
    }
}

fn sync_readonly_input(
    state: &Entity<InputState>,
    seed: &mut Option<SharedString>,
    value: Option<SharedString>,
    window: &mut Window,
    cx: &mut Context<AccountView>,
) {
    if seed != &value {
        let text = value.clone().unwrap_or_default();
        state.update(cx, |s, cx| s.set_value(text.to_string(), window, cx));
        *seed = value;
        return;
    }

    if let Some(value) = seed.as_ref() {
        if state.read(cx).value().as_ref() != value.as_ref() {
            let text = value.clone();
            state.update(cx, |s, cx| s.set_value(text.to_string(), window, cx));
        }
    }
}

fn account_credential_input(
    label: &'static str,
    id_prefix: &'static str,
    state: &Entity<InputState>,
    value: SharedString,
    show_secret_button: bool,
    secret_revealed: bool,
    on_toggle_secret: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let copy_value = value.clone();
    v_flex()
        .gap_1()
        .child(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(label),
        )
        .child(
            div()
                .id(SharedString::from(format!("{id_prefix}-input")))
                .probe_bounds(id_prefix, gpui::Role::TextInput, label)
                .w_full()
                .child(
                    Input::new(state).aria_label(label).suffix(
                        h_flex()
                            .gap_0p5()
                            .when(show_secret_button, |el| {
                                el.child(
                                    Button::new(SharedString::from(format!("{id_prefix}-show")))
                                        .ghost()
                                        .xsmall()
                                        .icon(if secret_revealed {
                                            IconName::EyeOff
                                        } else {
                                            IconName::Eye
                                        })
                                        .tooltip(if secret_revealed {
                                            "Hide account secret"
                                        } else {
                                            "Show account secret"
                                        })
                                        .on_click(on_toggle_secret),
                                )
                            })
                            .child(
                                Button::new(SharedString::from(format!("{id_prefix}-copy")))
                                    .ghost()
                                    .xsmall()
                                    .icon(IconName::Copy)
                                    .tooltip(format!("Copy {label}"))
                                    .on_click(move |_, _, cx| {
                                        cx.write_to_clipboard(ClipboardItem::new_string(
                                            copy_value.to_string(),
                                        ));
                                    }),
                            ),
                    ),
                ),
        )
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
    use super::{humanize_expiry, subscription_summary};

    const DAY: i64 = 24 * 60 * 60 * 1000;

    #[test]
    fn subscription_summary_reads_each_in_force_status() {
        let now = 1_900_000_000_000;
        assert_eq!(
            subscription_summary(Some("active"), None, now),
            "Your subscription is active."
        );
        assert!(
            subscription_summary(Some("trialing"), None, now).contains("trial"),
            "a trial should say so"
        );
        assert!(
            subscription_summary(Some("past_due"), None, now).contains("payment method"),
            "a past-due subscription should point at the fix"
        );
    }

    #[test]
    fn subscription_summary_adds_the_period_end_only_when_there_is_one() {
        let now = 1_900_000_000_000;
        assert_eq!(
            subscription_summary(Some("active"), Some(now + 12 * DAY), now),
            "Your subscription is active. The current billing period ends in 12d."
        );
        assert_eq!(
            subscription_summary(Some("active"), Some(now - DAY), now),
            "Your subscription is active. The current billing period has ended."
        );
    }

    #[test]
    fn an_unfamiliar_status_is_reported_rather_than_dressed_up() {
        let now = 1_900_000_000_000;
        let line = subscription_summary(Some("grace"), None, now);
        assert!(line.contains("grace"), "{line}");
    }

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
