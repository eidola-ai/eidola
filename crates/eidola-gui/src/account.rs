use gpui::{
    AppContext, Context, Entity, IntoElement, ParentElement, Render, SharedString, Styled,
    Subscription, Window, div,
};
use gpui_component::{
    ActiveTheme, Disableable, Sizable,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputState},
    label::Label,
    v_flex,
};

use crate::core::Core;

pub struct AccountView {
    core: Entity<Core>,
    allocate_state: Entity<InputState>,
    _subscriptions: Vec<Subscription>,
}

impl AccountView {
    pub fn new(core: Entity<Core>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let allocate_state = cx.new(|cx| InputState::new(window, cx).placeholder("Credits"));
        let _subscriptions = vec![cx.observe(&core, |_, _, cx| cx.notify())];

        // Trigger an initial price refresh so the section isn't empty.
        core.update(cx, |c, cx| {
            c.fetch_prices(cx);
            if c.config_state
                .as_ref()
                .map(|s| s.has_account)
                .unwrap_or(false)
            {
                c.fetch_balances(cx);
            }
        });

        Self {
            core,
            allocate_state,
            _subscriptions,
        }
    }
}

impl Render for AccountView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let core_ref = self.core.read(cx);
        let state = core_ref.config_state.as_ref();
        let has_account = state.map(|s| s.has_account).unwrap_or(false);

        let mut col = v_flex().p_4().gap_4().w_full();

        col = col.child(section_header("Account", cx));
        if has_account {
            col = col
                .child(
                    h_flex()
                        .gap_2()
                        .child(Label::new("Account configured").text_color(theme.success)),
                )
                .child(
                    Button::new("reset-account")
                        .danger()
                        .label("Reset Account")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.core.update(cx, |c, cx| c.reset_account(cx));
                        })),
                );
        } else {
            col = col
                .child(Label::new("No account").text_color(theme.muted_foreground))
                .child(
                    Button::new("create-account")
                        .primary()
                        .label("Create Account")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.core.update(cx, |c, cx| c.create_account(cx));
                        })),
                );
        }

        col = col.child(section_header("Balances", cx));
        if let Some(b) = core_ref.balances.as_ref() {
            col = col.child(
                h_flex()
                    .gap_2()
                    .child(Label::new("Available"))
                    .child(Label::new(SharedString::from(format!(
                        "{} credits",
                        b.available
                    )))),
            );
            for pool in &b.pools {
                let mut row = h_flex()
                    .gap_2()
                    .child(Label::new(SharedString::from(pool.source.clone())))
                    .child(Label::new(SharedString::from(format!(
                        "{} credits",
                        pool.amount
                    ))));
                if let Some(exp) = pool.expires_at.as_deref() {
                    row = row.child(
                        Label::new(SharedString::from(format!("expires {exp}")))
                            .text_color(theme.muted_foreground),
                    );
                }
                col = col.child(row);
            }
        } else {
            col = col.child(Label::new("Not loaded").text_color(theme.muted_foreground));
        }
        col = col.child(
            Button::new("refresh-balances")
                .label("Refresh Balances")
                .small()
                .disabled(!has_account)
                .on_click(cx.listener(|this, _, _, cx| {
                    this.core.update(cx, |c, cx| c.fetch_balances(cx));
                })),
        );

        col = col.child(section_header("Allocate Credits", cx));
        col = col.child(
            h_flex()
                .gap_2()
                .child(div().w_32().child(Input::new(&self.allocate_state)))
                .child(
                    Button::new("allocate")
                        .primary()
                        .label("Allocate")
                        .disabled(!has_account)
                        .on_click(cx.listener(|this, _, window, cx| {
                            let raw = this.allocate_state.read(cx).value().to_string();
                            if let Ok(amount) = raw.trim().parse::<i64>()
                                && amount > 0
                            {
                                this.core.update(cx, |c, cx| c.allocate_credits(amount, cx));
                                this.allocate_state.update(cx, |s, cx| {
                                    s.set_value("", window, cx);
                                });
                            }
                        })),
                ),
        );

        col = col.child(section_header("Available Plans", cx));
        if core_ref.prices.is_empty() {
            col = col.child(Label::new("No prices loaded").text_color(theme.muted_foreground));
        } else {
            for p in &core_ref.prices {
                let mut row = v_flex().gap_1().w_full();
                row = row.child(
                    h_flex()
                        .gap_2()
                        .justify_between()
                        .child(Label::new(SharedString::from(p.product_name.clone())))
                        .child(
                            Label::new(SharedString::from(format!(
                                "{}{}",
                                p.amount_display, p.recurrence
                            )))
                            .text_color(theme.muted_foreground),
                        ),
                );
                row = row.child(
                    Label::new(SharedString::from(format!("{} credits", p.credits)))
                        .text_color(theme.muted_foreground),
                );
                if let Some(desc) = p.product_description.as_deref() {
                    row = row.child(
                        Label::new(SharedString::from(desc.to_string()))
                            .text_color(theme.muted_foreground),
                    );
                }
                col = col.child(row);
            }
        }
        col = col.child(
            Button::new("refresh-prices")
                .label("Refresh Prices")
                .small()
                .on_click(cx.listener(|this, _, _, cx| {
                    this.core.update(cx, |c, cx| c.fetch_prices(cx));
                })),
        );

        if let Some(err) = core_ref.error_message.as_deref() {
            col = col.child(error_banner(err, cx));
        }

        col
    }
}

fn section_header(label: &str, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_color(theme.muted_foreground)
        .text_sm()
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
