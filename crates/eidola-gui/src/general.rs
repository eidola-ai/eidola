//! General settings pane — the server connection, honestly presented.
//!
//! The resting state is small: one Base URL row that says whether the value
//! is the trust-root **pin** baked into the binary or a user **override**
//! (with a one-click revert back to the pin). Everything else — attestation
//! URL, domain separator, hardware CAs, trusted measurements — is advanced
//! configuration that appears only while **⌥ is held**.
//!
//! The ⌥ state comes from the per-window `WindowInput` entity. `SettingsView`
//! (the root) is the one view that registers `on_modifiers_changed` and
//! mirrors events into it; `GeneralView` observes the entity here. This is
//! the fix for wave-2 bug 2: gpui dispatches `ModifiersChangedEvent` along
//! the focused element's ancestor path only, so a listener on this sibling
//! pane would be dead while a text input in the Account/Wallet pane (or the
//! Base URL field on this pane) has focus. Measurement rows summarize and
//! link to the Record window instead of dumping truncated hex.

use eidola_app_core::config::{AppearanceSetting, LightCharacter, TimeOfDayTint};
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
use crate::probe::Probe as _;
use crate::stores::ConfigStore;
use crate::window_input::WindowInput;

pub struct GeneralView {
    config: Entity<ConfigStore>,
    base_url_state: Entity<InputState>,
    /// Whether the Base URL row is in its edit state (input + save/cancel).
    editing_base_url: bool,
    /// Whether the ⌥-revealed advanced section is visible. Driven by the
    /// per-window `WindowInput` entity observed below; `set_advanced` is the
    /// single path (observer + behavior tests).
    advanced: bool,
    _subscriptions: Vec<Subscription>,
}

impl GeneralView {
    /// `window_input` is the per-window modifier entity owned by
    /// `SettingsView`. This view observes it so ⌥ transitions fire
    /// `set_advanced` regardless of which pane or input has focus in the
    /// window — the fix for wave-2 bug 2. `GeneralView` never registers its
    /// own `on_modifiers_changed` listener.
    pub fn new(
        config: Entity<ConfigStore>,
        window_input: Entity<WindowInput>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let initial = config
            .read(cx)
            .eidola_trust()
            .map(|t| t.base_url.clone())
            .unwrap_or_default();

        let base_url_state = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("https://…")
                .default_value(&initial)
        });

        let _subscriptions = vec![
            cx.observe(&config, |_, _, cx| cx.notify()),
            // Mirror ⌥ transitions into the advanced flag. The observer fires
            // whenever `WindowInput` emits (on every modifier change), so this
            // is always in sync with the root's listener — even while a text
            // input in a sibling pane has focus.
            cx.observe(&window_input, |this: &mut Self, wi, cx| {
                let alt = wi.read(cx).alt_held();
                this.set_advanced(alt, cx);
            }),
        ];

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
            .eidola_trust()
            .map(|t| t.base_url.clone())
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
        let pin = self
            .config
            .read(cx)
            .eidola_trust()
            .map(|t| t.base_url_pin.clone());
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

    /// Circadian day/night axis. Writes through the store; the theme
    /// re-applies via its config observation (`theme::wire_config`).
    pub fn set_appearance(&mut self, appearance: AppearanceSetting, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.set_appearance(appearance, cx));
        cx.notify();
    }

    /// Circadian time-of-day axis.
    pub fn set_time_of_day_tint(&mut self, tint: TimeOfDayTint, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.set_time_of_day_tint(tint, cx));
        cx.notify();
    }

    /// Circadian fixed light character (shown only while the time-of-day
    /// axis is off).
    pub fn set_light_character(&mut self, character: LightCharacter, cx: &mut Context<Self>) {
        self.config
            .update(cx, |c, cx| c.set_light_character(character, cx));
        cx.notify();
    }
}

impl Render for GeneralView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let store = self.config.read(cx);
        let state = store.state().cloned();
        let trust = store.eidola_trust().cloned();
        let error = store.error().map(|e| e.to_string());

        // Note: ⌥ state is driven by the `WindowInput` observer installed in
        // `new`; there is no `on_modifiers_changed` listener here. See the
        // module doc for why the listener lives on the `SettingsView` root.
        let mut col = v_flex().id("general-pane").px_6().py_5().gap_4().w_full();

        // --- Appearance: the circadian theme's two axes -----------------
        col = col.child(section_header("Appearance", cx));

        if let Some(s) = state.as_ref() {
            let mut day_night = h_flex().gap_2();
            for (value, id, probe_name, label) in [
                (
                    AppearanceSetting::Auto,
                    "appearance-auto",
                    "settings/general/appearance/auto",
                    "Auto",
                ),
                (
                    AppearanceSetting::System,
                    "appearance-system",
                    "settings/general/appearance/system",
                    "System",
                ),
                (
                    AppearanceSetting::Day,
                    "appearance-day",
                    "settings/general/appearance/day",
                    "Day",
                ),
                (
                    AppearanceSetting::Night,
                    "appearance-night",
                    "settings/general/appearance/night",
                    "Night",
                ),
            ] {
                let active = s.appearance == value;
                day_night = day_night.child(
                    choice_chip(id, label, active, cx)
                        .probe(probe_name, gpui::Role::Button, label)
                        .aria_selected(active)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.set_appearance(value, cx);
                        })),
                );
            }
            col = col.child(field_row(
                "Day & night",
                cx,
                v_flex().gap_1().child(day_night).child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground.opacity(0.8))
                        .child("System follows macOS. Auto follows the sun — day between sunrise and sunset."),
                ),
            ));

            let mut tint_row = h_flex().gap_2();
            for (value, id, probe_name, label) in [
                (
                    TimeOfDayTint::On,
                    "time-of-day-on",
                    "settings/general/time-of-day/on",
                    "On",
                ),
                (
                    TimeOfDayTint::Off,
                    "time-of-day-off",
                    "settings/general/time-of-day/off",
                    "Off",
                ),
            ] {
                let active = s.time_of_day_tint == value;
                tint_row = tint_row.child(
                    choice_chip(id, label, active, cx)
                        .probe(probe_name, gpui::Role::Button, label)
                        .aria_selected(active)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.set_time_of_day_tint(value, cx);
                        })),
                );
            }
            col = col.child(field_row(
                "Time of day",
                cx,
                v_flex().gap_1().child(tint_row).child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground.opacity(0.8))
                        .child(
                            "Follows the sun's light — cool around sunrise, neutral midday and \
                             midnight, warm around sunset. Sunrise and sunset are approximated \
                             from your time zone.",
                        ),
                ),
            ));

            // With the sun turned off, the character becomes a fixed choice.
            if s.time_of_day_tint == TimeOfDayTint::Off {
                let mut light_row = h_flex().gap_2();
                for (value, id, probe_name, label) in [
                    (
                        LightCharacter::Cool,
                        "light-character-cool",
                        "settings/general/light-character/cool",
                        "Cool",
                    ),
                    (
                        LightCharacter::Neutral,
                        "light-character-neutral",
                        "settings/general/light-character/neutral",
                        "Neutral",
                    ),
                    (
                        LightCharacter::Warm,
                        "light-character-warm",
                        "settings/general/light-character/warm",
                        "Warm",
                    ),
                ] {
                    let active = s.light_character == value;
                    light_row = light_row.child(
                        choice_chip(id, label, active, cx)
                            .probe(probe_name, gpui::Role::Button, label)
                            .aria_selected(active)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.set_light_character(value, cx);
                            })),
                    );
                }
                col = col.child(field_row(
                    "Light",
                    cx,
                    v_flex().gap_1().child(light_row).child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground.opacity(0.8))
                            .child("A fixed light character for the palette."),
                    ),
                ));
            }
        }

        col = col.child(div().pt_2().child(section_header("Server", cx)));

        // --- Base URL: honest about override vs pin --------------------
        let mut base_value = v_flex().flex_1().gap_1();
        if self.editing_base_url {
            base_value = base_value
                .child(
                    // Probed wrapper for the a11y role/label — probe the
                    // wrapping div, not the gpui-component Input.
                    div()
                        .id("base-url-input-wrap")
                        .probe(
                            "settings/general/base-url",
                            gpui::Role::TextInput,
                            "Base URL",
                        )
                        .w_full()
                        .flex()
                        .child(Input::new(&self.base_url_state).flex_1()),
                )
                .child(
                    h_flex()
                        .gap_2()
                        .pt_1()
                        .child(
                            // Probed wrapper for the a11y role/label — shrink-wraps
                            // the button so its bounds are an honest click target.
                            div()
                                .id("save-base-url-wrap")
                                .probe("settings/general/save", gpui::Role::Button, "Save")
                                .child(
                                    Button::new("save-base-url")
                                        .primary()
                                        .small()
                                        .label("Save")
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.save_base_url(cx)),
                                        ),
                                ),
                        )
                        .child(
                            // Probed wrapper for the a11y role/label — shrink-wraps
                            // the button so its bounds are an honest click target.
                            div()
                                .id("cancel-base-url-wrap")
                                .probe("settings/general/cancel", gpui::Role::Button, "Cancel")
                                .child(
                                    Button::new("cancel-base-url")
                                        .ghost()
                                        .small()
                                        .label("Cancel")
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.cancel_edit_base_url(cx)
                                        })),
                                ),
                        ),
                );
        } else if let Some(s) = trust.as_ref() {
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
                        .probe(
                            "settings/general/revert-to-pin",
                            gpui::Role::Button,
                            "Revert to pin",
                        )
                        .on_click(cx.listener(|this, _, _, cx| this.revert_base_url(cx))),
                );
            }
            links = links.child(
                quiet_link("edit-base-url", "Change…", cx)
                    .probe(
                        "settings/general/change",
                        gpui::Role::Button,
                        "Change base URL",
                    )
                    .on_click(
                        cx.listener(|this, _, window, cx| this.begin_edit_base_url(window, cx)),
                    ),
            );
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

                // The hardware CAs + trusted-measurements summary come from
                // the eidola backend row (the connection + trust bundle),
                // not `ConfigState`.
                let (has_root_ca, has_intermediate_ca) = trust
                    .as_ref()
                    .map(|t| (t.has_hardware_root_ca, t.has_hardware_intermediate_ca))
                    .unwrap_or((false, false));
                col = col.child(field_row(
                    "Hardware root CA",
                    cx,
                    muted_text(
                        if has_root_ca {
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
                        if has_intermediate_ca {
                            "Custom certificate set"
                        } else {
                            "Not set — AMD/Intel vendor chain"
                        },
                        cx,
                    ),
                ));

                // Measurements: a summary + a door, never a hex dump.
                let measurements_are_override = trust
                    .as_ref()
                    .map(|t| t.trusted_measurements_are_override)
                    .unwrap_or(false);
                let measurements_len = trust
                    .as_ref()
                    .map(|t| t.trusted_measurements.len())
                    .unwrap_or(0);
                let summary = if measurements_are_override {
                    format!(
                        "{} user-trusted measurement{}",
                        measurements_len,
                        if measurements_len == 1 { "" } else { "s" }
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
                                format!(
                                    "Inspect attestation evidence in the Record ({})",
                                    crate::actions::primary_shift_chord("L")
                                ),
                                cx,
                            )
                            .probe(
                                "settings/general/open-record",
                                gpui::Role::Link,
                                "Inspect attestation evidence in the Record",
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
                    .child(format!(
                        "Hold {} for advanced configuration.",
                        crate::actions::alt_name()
                    )),
            );
        }

        if let Some(err) = error {
            col = col.child(
                div()
                    .id("general-error")
                    .probe("settings/general/error", gpui::Role::Alert, err.clone())
                    .child(error_banner(&err, cx)),
            );
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

/// One selectable option in a small chip row (the appearance settings).
/// The active chip gets the sidebar-accent pill, matching the settings nav.
fn choice_chip(
    id: &'static str,
    label: &'static str,
    active: bool,
    cx: &gpui::App,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    let el = div()
        .id(id)
        .cursor_pointer()
        .px_2()
        .py_0p5()
        .rounded_md()
        .text_sm();
    let el = if active {
        el.bg(theme.sidebar_accent)
            .text_color(theme.sidebar_accent_foreground)
    } else {
        el.text_color(theme.muted_foreground)
            .hover(|s| s.text_color(theme.foreground))
    };
    el.child(label)
}

/// A quiet inline text link: muted, brightening on hover. The settings
/// surface's only interaction affordance besides explicit buttons.
fn quiet_link(
    id: &'static str,
    label: impl Into<gpui::SharedString>,
    cx: &gpui::App,
) -> gpui::Stateful<gpui::Div> {
    let theme = cx.theme();
    div()
        .id(id)
        .cursor_pointer()
        .text_color(theme.link)
        .hover(|s| s.text_color(theme.link_hover))
        .child(label.into())
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
