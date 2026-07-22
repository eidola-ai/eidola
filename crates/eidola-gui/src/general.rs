//! General settings pane — how the app looks.
//!
//! The whole pane is the circadian **Appearance** axes (day/night, time of
//! day, and the fixed light character while the sun is off). Everything
//! about the Eidola *connection* — base URL, trusted measurements, hardware
//! CAs, attestation URL, domain separator — lives in Settings → Backends →
//! Eidola, the eidola backend's own configuration surface; this pane no
//! longer summarizes or duplicates any of it. (Earlier iterations kept
//! read-only trust summaries here, first behind a ⌥-hold reveal and then a
//! click-to-expand disclosure; both duplicated the Backends surface and are
//! gone.)

use eidola_app_core::config::{AppearanceSetting, LightCharacter, TimeOfDayTint};
use gpui::{
    Context, Entity, InteractiveElement, IntoElement, ParentElement, Render, SharedString,
    StatefulInteractiveElement, Styled, Subscription, Window, div,
};
use gpui_component::{ActiveTheme, StyledExt, h_flex, label::Label, v_flex};

use crate::probe::Probe as _;
use crate::stores::ConfigStore;

pub struct GeneralView {
    config: Entity<ConfigStore>,
    _subscriptions: Vec<Subscription>,
}

impl GeneralView {
    pub fn new(config: Entity<ConfigStore>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let _subscriptions = vec![cx.observe(&config, |_, _, cx| cx.notify())];

        Self {
            config,
            _subscriptions,
        }
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
        let error = store.error().map(|e| e.to_string());

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
