use gpui::{
    AppContext, Context, Entity, IntoElement, ParentElement, Render, SharedString, Styled,
    Subscription, Window, div,
};
use gpui_component::{
    ActiveTheme, Disableable, Sizable, StyledExt,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputState},
    label::Label,
    v_flex,
};

use crate::core::Core;

pub struct GeneralView {
    core: Entity<Core>,
    base_url_state: Entity<InputState>,
    _subscriptions: Vec<Subscription>,
}

impl GeneralView {
    pub fn new(core: Entity<Core>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let initial = core
            .read(cx)
            .config_state
            .as_ref()
            .map(|s| s.base_url.clone())
            .unwrap_or_default();

        let base_url_state = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("https://…")
                .default_value(&initial)
        });

        let _subscriptions = vec![cx.observe(&core, |_, _, cx| cx.notify())];

        Self {
            core,
            base_url_state,
            _subscriptions,
        }
    }

    fn save_base_url(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let value = self.base_url_state.read(cx).value().to_string();
        if value.trim().is_empty() {
            return;
        }
        self.core
            .update(cx, |core, cx| core.set_base_url(value, cx));
    }
}

impl Render for GeneralView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let core = self.core.read(cx);
        let state_clone = core.config_state.as_ref().map(|s| {
            (
                s.attestation_url.clone(),
                s.trusted_measurements
                    .iter()
                    .map(|m| (m.snp.clone(), m.tdx_rtmr1.clone(), m.tdx_rtmr2.clone()))
                    .collect::<Vec<_>>(),
                s.has_hardware_root_ca,
                s.has_hardware_intermediate_ca,
                s.domain_separator.clone(),
            )
        });
        let error = core.error_message.clone();

        let mut col = v_flex().p_4().gap_4().w_full();

        col = col.child(section_header("Server", cx));
        col = col.child(field_row(
            "Base URL",
            cx,
            h_flex()
                .gap_2()
                .child(div().flex_1().child(Input::new(&self.base_url_state)))
                .child(
                    Button::new("save-base-url")
                        .primary()
                        .label("Save")
                        .small()
                        .disabled(false)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.save_base_url(window, cx);
                        })),
                ),
        ));

        if let Some((attest_url, measurements, has_root_ca, has_int_ca, domain_separator)) =
            state_clone
        {
            col = col.child(section_header("Attestation", cx));
            col = col.child(field_row(
                "Attestation URL",
                cx,
                muted_text(
                    attest_url.unwrap_or_else(|| "Default (Tinfoil ATC)".into()),
                    cx,
                ),
            ));
            col = col.child(field_row(
                "Trusted measurements",
                cx,
                muted_text(
                    if measurements.is_empty() {
                        "None (attestation disabled)".to_string()
                    } else {
                        format!("{} measurement(s)", measurements.len())
                    },
                    cx,
                ),
            ));

            for (snp, rtmr1, rtmr2) in &measurements {
                col = col.child(
                    v_flex()
                        .pl_4()
                        .gap_1()
                        .text_color(theme.muted_foreground)
                        .text_xs()
                        .child(SharedString::from(format!("SNP: {}…", short(snp))))
                        .child(SharedString::from(format!("RTMR1: {}…", short(rtmr1))))
                        .child(SharedString::from(format!("RTMR2: {}…", short(rtmr2)))),
                );
            }

            col = col.child(field_row(
                "Hardware Root CA",
                cx,
                muted_text(if has_root_ca { "Set" } else { "Not set" }, cx),
            ));
            col = col.child(field_row(
                "Hardware Intermediate CA",
                cx,
                muted_text(if has_int_ca { "Set" } else { "Not set" }, cx),
            ));

            col = col.child(section_header("Protocol", cx));
            col = col.child(field_row(
                "Domain Separator",
                cx,
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(domain_separator)),
            ));
        }

        if let Some(err) = error {
            col = col.child(error_banner(&err, cx));
        }

        col
    }
}

fn short(s: &str) -> String {
    s.chars().take(32).collect()
}

fn section_header(label: &str, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_color(theme.muted_foreground)
        .text_sm()
        .font_medium()
        .child(SharedString::from(label.to_string()))
}

fn field_row<C: IntoElement>(label: &str, cx: &gpui::App, child: C) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .w_full()
        .gap_4()
        .py_1()
        .child(
            div()
                .w_40()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(label.to_string())),
        )
        .child(div().flex_1().child(child))
}

fn muted_text(text: impl Into<String>, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    let text = text.into();
    div()
        .text_color(theme.muted_foreground)
        .child(SharedString::from(text))
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
