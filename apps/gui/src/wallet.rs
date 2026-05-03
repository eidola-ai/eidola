use gpui::{
    Context, Entity, IntoElement, ParentElement, Render, SharedString, Styled, Subscription,
    Window, div,
};
use gpui_component::{ActiveTheme, button::Button, h_flex, label::Label, v_flex};

use crate::core::Core;

pub struct WalletView {
    core: Entity<Core>,
    _subscriptions: Vec<Subscription>,
}

impl WalletView {
    pub fn new(core: Entity<Core>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let _subscriptions = vec![cx.observe(&core, |_, _, cx| cx.notify())];
        core.update(cx, |c, cx| c.fetch_credentials(cx));
        Self {
            core,
            _subscriptions,
        }
    }
}

impl Render for WalletView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let core_ref = self.core.read(cx);

        let mut col = v_flex().p_4().gap_3().w_full();

        col = col.child(
            h_flex()
                .justify_between()
                .child(Label::new("Active Credentials").text_color(theme.muted_foreground))
                .child(
                    Button::new("refresh-credentials")
                        .label("Refresh")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.core.update(cx, |c, cx| c.fetch_credentials(cx));
                        })),
                ),
        );

        if core_ref.credentials.is_empty() {
            col = col.child(
                v_flex()
                    .gap_1()
                    .py_8()
                    .items_center()
                    .child(Label::new("No Credentials").text_color(theme.muted_foreground))
                    .child(
                        Label::new("Allocate credits from Account to get started.")
                            .text_color(theme.muted_foreground),
                    ),
            );
        } else {
            for cred in &core_ref.credentials {
                let nonce_short: String = cred.nonce.chars().take(16).collect();
                col = col.child(
                    h_flex()
                        .w_full()
                        .py_2()
                        .gap_3()
                        .border_b_1()
                        .border_color(theme.border)
                        .child(
                            v_flex()
                                .flex_1()
                                .gap_1()
                                .child(div().child(SharedString::from(format!("{nonce_short}…"))))
                                .child(
                                    Label::new(SharedString::from(format!(
                                        "Generation {}",
                                        cred.generation
                                    )))
                                    .text_color(theme.muted_foreground),
                                ),
                        )
                        .child(Label::new(SharedString::from(format!(
                            "{} credits",
                            cred.credits
                        )))),
                );
            }
        }

        col
    }
}
