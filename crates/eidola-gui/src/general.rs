//! General settings pane — how the app looks, and when it starts.
//!
//! Mostly the circadian **Appearance** axes (day/night, time of day, and the
//! fixed light character while the sun is off), plus the one **Startup**
//! row: "Open at login" (task 17 wave 3), opt-in and never default. That row
//! is the only state in this pane that does not come from a store — the
//! system owns it (see [`crate::login_item`]), so it is read at construction
//! and re-read after every write. Everything
//! about the Eidola *connection* — base URL, trusted measurements, hardware
//! CAs, domain separator — lives in Settings → Backends →
//! Eidola, the eidola backend's own configuration surface; this pane no
//! longer summarizes or duplicates any of it. (Earlier iterations kept
//! read-only trust summaries here, first behind a ⌥-hold reveal and then a
//! click-to-expand disclosure; both duplicated the Backends surface and are
//! gone.)

use eidola_app_core::config::{AppearanceSetting, LightCharacter, TimeOfDayTint};
use gpui::{
    Context, Entity, InteractiveElement, IntoElement, ParentElement, Render, SharedString,
    StatefulInteractiveElement, Styled, Subscription, Window, div, prelude::FluentBuilder,
};
use gpui_component::{
    ActiveTheme, Sizable, StyledExt, h_flex, label::Label, switch::Switch, v_flex,
};

use crate::login_item::{self, LoginItemState};
use crate::probe::Probe as _;
use crate::stores::ConfigStore;

pub struct GeneralView {
    config: Entity<ConfigStore>,
    /// The system's answer about our login item. Not store-backed — macOS
    /// owns this state and a second copy would be a second answer (see
    /// `crate::login_item`). Read once here and again after every write.
    login_item: LoginItemState,
    /// The system's own words when it refuses a register/unregister, until
    /// the next attempt.
    login_item_error: Option<String>,
    _subscriptions: Vec<Subscription>,
}

impl GeneralView {
    pub fn new(config: Entity<ConfigStore>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let _subscriptions = vec![cx.observe(&config, |_, _, cx| cx.notify())];

        Self {
            config,
            login_item: login_item::state(),
            login_item_error: None,
            _subscriptions,
        }
    }

    /// The login item as the system last reported it (test accessor).
    pub fn login_item(&self) -> LoginItemState {
        self.login_item
    }

    pub fn login_item_error(&self) -> Option<&str> {
        self.login_item_error.as_deref()
    }

    /// Turn "Open at login" on or off. **The write's result is never
    /// assumed** — a refusal (unsigned bundle, a user-level denial) leaves
    /// the system exactly as it was, so the row re-reads the system rather
    /// than flipping to what was asked for.
    pub fn set_open_at_login(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.login_item_error = login_item::set(enabled).err();
        self.login_item = login_item::state();
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

    /// Text size — the same type-scale ladder as View → Zoom In / Zoom Out /
    /// Actual Size, surfaced here as a visible control (the menu shortcuts stay
    /// the fast path, but a settings row is where an older user looks first).
    pub fn zoom_in(&mut self, cx: &mut Context<Self>) {
        self.config.update(cx, |c, cx| c.zoom_in(cx));
        cx.notify();
    }

    pub fn zoom_out(&mut self, cx: &mut Context<Self>) {
        self.config.update(cx, |c, cx| c.zoom_out(cx));
        cx.notify();
    }

    pub fn reset_zoom(&mut self, cx: &mut Context<Self>) {
        self.config.update(cx, |c, cx| c.reset_zoom(cx));
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
                        // The chips read "Auto"/"System"/"Day"/"Night"; the
                        // group label that gives them meaning is a plain,
                        // node-less `div`, so the name has to carry it.
                        .probe(
                            probe_name,
                            gpui::Role::Button,
                            format!("Day & night: {label}"),
                        )
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
                        .probe(
                            probe_name,
                            gpui::Role::Button,
                            format!("Time of day: {label}"),
                        )
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
                            .probe(probe_name, gpui::Role::Button, format!("Light: {label}"))
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

            // --- Text size: the type-scale ladder (also View → Zoom …) -------
            let at_min = s.font_scale <= eidola_app_core::config::FONT_SCALE_MIN + 1e-3;
            let at_max = s.font_scale >= eidola_app_core::config::FONT_SCALE_MAX - 1e-3;
            let percent = format!("{}%", (s.font_scale * 100.0).round() as i32);
            let size_row = h_flex()
                .gap_2()
                .items_center()
                .child(
                    choice_chip("text-size-smaller", "A−", false, cx)
                        // The current scale renders as a bare `div` with no
                        // node of its own, so the chips carry it — otherwise
                        // "smaller" is announced with nothing to be smaller
                        // than.
                        .probe(
                            "settings/general/text-size/smaller",
                            gpui::Role::Button,
                            format!("Smaller text, currently {percent}"),
                        )
                        .when(at_min, |el| el.opacity(0.4))
                        .on_click(cx.listener(|this, _, _, cx| this.zoom_out(cx))),
                )
                .child(
                    choice_chip("text-size-larger", "A+", false, cx)
                        .probe(
                            "settings/general/text-size/larger",
                            gpui::Role::Button,
                            format!("Larger text, currently {percent}"),
                        )
                        .when(at_max, |el| el.opacity(0.4))
                        .on_click(cx.listener(|this, _, _, cx| this.zoom_in(cx))),
                )
                .child(
                    choice_chip("text-size-reset", "Reset", false, cx)
                        .probe(
                            "settings/general/text-size/reset",
                            gpui::Role::Button,
                            "Reset text size to 100%",
                        )
                        .on_click(cx.listener(|this, _, _, cx| this.reset_zoom(cx))),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(percent)),
                );
            col = col.child(field_row(
                "Text size",
                cx,
                v_flex().gap_1().child(size_row).child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground.opacity(0.8))
                        .child(
                            "Scales all text. Also on the View menu — Actual Size, Zoom In, \
                             Zoom Out.",
                        ),
                ),
            ));
        }

        // --- Startup: the opt-in login item -----------------------------
        col = col.child(section_header("Startup", cx));
        col = col.child(field_row(
            "Open at login",
            cx,
            v_flex()
                .gap_1()
                .child(self.login_item_toggle(cx))
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground.opacity(0.8))
                        .child(self.login_item.description()),
                )
                .when_some(self.login_item_error.clone(), |el, err| {
                    el.child(
                        div()
                            .id("login-item-error")
                            .probe(
                                "settings/general/login-item/error",
                                gpui::Role::Alert,
                                err.clone(),
                            )
                            .child(error_banner(&err, cx)),
                    )
                }),
        ));

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

impl GeneralView {
    /// The "Open at login" switch. Same hoisted shape as the backends pane's
    /// auto-start toggle: the probed wrapper is the accessible control (role,
    /// label, toggled state, keyboard activation) because `Switch` tracks no
    /// focus handle at our gpui-component rev, and `Switch` handles the
    /// pointer press itself with `stop_propagation`, so the two never
    /// double-fire.
    fn login_item_toggle(&self, cx: &Context<Self>) -> impl IntoElement + use<> {
        let state = self.login_item;
        let next = !state.is_on();
        let settable = state.is_settable();
        div()
            .id("login-item")
            .probe(
                "settings/general/login-item",
                gpui::Role::CheckBox,
                "Open Eidola at login",
            )
            // `aria_toggled`, not `aria_selected`: `accesskit_macos` reads a
            // checkbox's value from `toggled()` and consults `is_selected`
            // only for `Role::Tab`.
            .aria_toggled(state.is_on().into())
            // One predicate for activation *and* tab-stopness — a wrapper
            // that keeps its click while dropping only the Tab entry is a
            // control that lies to the pointer.
            .map(|el| {
                if settable {
                    el.on_click(cx.listener(move |this, _, _, cx| {
                        this.set_open_at_login(next, cx);
                    }))
                } else {
                    el.tab_stop(false).opacity(0.5)
                }
            })
            .child(
                // Switch sets no AccessKit role/label at our gpui-component
                // rev, so the probed wrapper is the only node; if Switch
                // gains self-annotation upstream, this site must join the
                // `.role(None)` opt-out.
                Switch::new("login-item-switch")
                    .small()
                    .checked(state.is_on())
                    .label("Start Eidola when you log in")
                    .when(settable, |s| {
                        s.on_click(cx.listener(move |this, checked: &bool, _, cx| {
                            this.set_open_at_login(*checked, cx);
                        }))
                    }),
            )
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
