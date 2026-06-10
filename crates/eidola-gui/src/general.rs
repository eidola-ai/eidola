//! General settings pane — the server connection, honestly presented.
//!
//! The resting state is small: one Base URL row that says whether the value
//! is the trust-root **pin** baked into the binary or a user **override**
//! (with a one-click revert back to the pin). Everything else — attestation
//! URL, domain separator, hardware CAs, trusted measurements — is advanced
//! configuration that appears only while **⌥ is held** (tracked via
//! `on_modifiers_changed` on the pane root; gpui delivers modifier changes
//! to every painted element that registers, so no focus gymnastics needed).
//! Measurement rows summarize and link to the Record window instead of
//! dumping truncated hex.

use gpui::{
    AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, Subscription, Window, div,
};
use gpui_component::{
    ActiveTheme, Sizable, StyledExt,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputState},
    label::Label,
    v_flex,
};

use crate::actions::OpenRecord;
use crate::stores::ConfigStore;

pub struct GeneralView {
    config: Entity<ConfigStore>,
    base_url_state: Entity<InputState>,
    /// Whether the Base URL row is in its edit state (input + save/cancel).
    editing_base_url: bool,
    /// Whether the ⌥-revealed advanced section is visible. Mirrors the live
    /// Option-key state via `on_modifiers_changed`.
    advanced: bool,
    _subscriptions: Vec<Subscription>,
}

impl GeneralView {
    pub fn new(config: Entity<ConfigStore>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let initial = config
            .read(cx)
            .state()
            .map(|s| s.base_url.clone())
            .unwrap_or_default();

        let base_url_state = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("https://…")
                .default_value(&initial)
        });

        let _subscriptions = vec![cx.observe(&config, |_, _, cx| cx.notify())];

        Self {
            config,
            base_url_state,
            editing_base_url: false,
            advanced: false,
            _subscriptions,
        }
    }

    /// Set the advanced (⌥-revealed) state. Public so the modifiers listener
    /// and behavior tests share one path.
    pub fn set_advanced(&mut self, on: bool, cx: &mut Context<Self>) {
        if self.advanced != on {
            self.advanced = on;
            cx.notify();
        }
    }

    pub fn advanced(&self) -> bool {
        self.advanced
    }

    pub fn editing_base_url(&self) -> bool {
        self.editing_base_url
    }

    /// Enter the Base URL edit state, seeding the input with the current
    /// resolved value.
    pub fn begin_edit_base_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let current = self
            .config
            .read(cx)
            .state()
            .map(|s| s.base_url.clone())
            .unwrap_or_default();
        self.base_url_state.update(cx, |s, cx| {
            s.set_value(&current, window, cx);
        });
        self.editing_base_url = true;
        cx.notify();
    }

    pub fn cancel_edit_base_url(&mut self, cx: &mut Context<Self>) {
        self.editing_base_url = false;
        cx.notify();
    }

    /// Save the edited value as an override. Saving the pin itself is
    /// treated as a revert — the config stays honest about its source.
    pub fn save_base_url(&mut self, cx: &mut Context<Self>) {
        let value = self.base_url_state.read(cx).value().trim().to_string();
        if value.is_empty() {
            return;
        }
        let pin = self.config.read(cx).state().map(|s| s.base_url_pin.clone());
        self.config.update(cx, |c, cx| {
            if pin.as_deref() == Some(value.as_str()) {
                c.clear_base_url_override(cx);
            } else {
                c.set_base_url(value, cx);
            }
        });
        self.editing_base_url = false;
        cx.notify();
    }

    /// One-click revert from an override back to the built-in pin.
    pub fn revert_base_url(&mut self, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.clear_base_url_override(cx));
        self.editing_base_url = false;
        cx.notify();
    }
}

impl Render for GeneralView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let store = self.config.read(cx);
        let state = store.state().cloned();
        let error = store.error().map(|e| e.to_string());

        let mut col = v_flex()
            .id("general-pane")
            .px_6()
            .py_5()
            .gap_4()
            .w_full()
            .on_modifiers_changed(cx.listener(|this, e: &gpui::ModifiersChangedEvent, _, cx| {
                this.set_advanced(e.modifiers.alt, cx);
            }));

        col = col.child(section_header("Server", cx));

        // --- Base URL: honest about override vs pin --------------------
        let mut base_value = v_flex().flex_1().gap_1();
        if self.editing_base_url {
            base_value = base_value.child(Input::new(&self.base_url_state)).child(
                h_flex()
                    .gap_2()
                    .pt_1()
                    .child(
                        Button::new("save-base-url")
                            .primary()
                            .small()
                            .label("Save")
                            .on_click(cx.listener(|this, _, _, cx| this.save_base_url(cx))),
                    )
                    .child(
                        Button::new("cancel-base-url")
                            .ghost()
                            .small()
                            .label("Cancel")
                            .on_click(cx.listener(|this, _, _, cx| this.cancel_edit_base_url(cx))),
                    ),
            );
        } else if let Some(s) = state.as_ref() {
            base_value = base_value.child(
                div()
                    .text_sm()
                    .font_family("Menlo")
                    .child(SharedString::from(s.base_url.clone())),
            );
            // Status sentence in its own full-width line so it wraps; the
            // quiet links sit on the line below.
            let status: String = if s.base_url_is_override {
                format!("Override — the built-in pin is {}.", s.base_url_pin)
            } else {
                "Built-in pin — verified against this build's trust root.".into()
            };
            base_value = base_value.child(
                div()
                    .w_full()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(status)),
            );
            let mut links = h_flex().gap_3().text_xs();
            if s.base_url_is_override {
                links = links.child(
                    quiet_link("revert-base-url", "Revert to pin", cx)
                        .on_click(cx.listener(|this, _, _, cx| this.revert_base_url(cx))),
                );
            }
            links =
                links.child(quiet_link("edit-base-url", "Change…", cx).on_click(
                    cx.listener(|this, _, window, cx| this.begin_edit_base_url(window, cx)),
                ));
            base_value = base_value.child(links);
        }
        col = col.child(field_row("Base URL", cx, base_value));

        // --- Advanced (⌥-revealed) --------------------------------------
        if self.advanced {
            if let Some(s) = state.as_ref() {
                col = col.child(div().pt_2().child(section_header("Advanced", cx)));

                col = col.child(field_row(
                    "Attestation URL",
                    cx,
                    muted_text(
                        s.attestation_url
                            .clone()
                            .unwrap_or_else(|| "Default (Tinfoil ATC)".into()),
                        cx,
                    ),
                ));

                // The domain separator is one long unbreakable token, so it
                // gets a stacked row (value under label, full width) rather
                // than the two-column layout.
                col = col.child(
                    v_flex()
                        .w_full()
                        .py_1()
                        .gap_1()
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.muted_foreground)
                                .child("Domain separator"),
                        )
                        .child(
                            div()
                                .text_xs()
                                .font_family("Menlo")
                                .text_color(theme.muted_foreground)
                                .child(SharedString::from(s.domain_separator.clone())),
                        ),
                );

                col = col.child(field_row(
                    "Hardware root CA",
                    cx,
                    muted_text(
                        if s.has_hardware_root_ca {
                            "Custom certificate set"
                        } else {
                            "Not set — AMD/Intel vendor chain"
                        },
                        cx,
                    ),
                ));
                col = col.child(field_row(
                    "Intermediate CA",
                    cx,
                    muted_text(
                        if s.has_hardware_intermediate_ca {
                            "Custom certificate set"
                        } else {
                            "Not set — AMD/Intel vendor chain"
                        },
                        cx,
                    ),
                ));

                // Measurements: a summary + a door, never a hex dump.
                let summary = if s.trusted_measurements_are_override {
                    format!(
                        "{} user-trusted measurement{}",
                        s.trusted_measurements.len(),
                        if s.trusted_measurements.len() == 1 {
                            ""
                        } else {
                            "s"
                        }
                    )
                } else {
                    "1 measurement — pinned at build".to_string()
                };
                col = col.child(field_row(
                    "Trusted measurements",
                    cx,
                    v_flex().gap_1().child(muted_text(summary, cx)).child(
                        h_flex().text_xs().text_color(theme.muted_foreground).child(
                            quiet_link(
                                "open-record",
                                "Inspect attestation evidence in the Record (⇧⌘L)",
                                cx,
                            )
                            .on_click(|_, window, cx| {
                                window.dispatch_action(Box::new(OpenRecord), cx);
                            }),
                        ),
                    ),
                ));
            }
        } else {
            // One quiet line of disclosure so the ⌥ affordance is
            // discoverable without a persistent "Advanced" section.
            col = col.child(
                div()
                    .pt_2()
                    .text_xs()
                    .italic()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    .child("Hold ⌥ for advanced configuration."),
            );
        }

        if let Some(err) = error {
            col = col.child(error_banner(&err, cx));
        }

        col
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

fn field_row<C: IntoElement>(label: &str, cx: &gpui::App, child: C) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .w_full()
        .gap_4()
        .py_1()
        .items_start()
        .child(
            div()
                .w(gpui::px(144.))
                .flex_none()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(label.to_string())),
        )
        .child(div().flex_1().min_w_0().child(child))
}

fn muted_text(text: impl Into<String>, cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    let text = text.into();
    div()
        .text_color(theme.muted_foreground)
        .child(SharedString::from(text))
}

/// A quiet inline text link: muted, brightening on hover. The settings
/// surface's only interaction affordance besides explicit buttons.
fn quiet_link(id: &'static str, label: &'static str, cx: &gpui::App) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    div()
        .id(id)
        .cursor_pointer()
        .text_color(theme.link)
        .hover(|s| s.text_color(theme.link_hover))
        .child(label)
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
