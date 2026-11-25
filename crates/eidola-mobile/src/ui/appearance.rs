//! Appearance: the desktop's Settings ▸ General appearance axes that apply on
//! a phone — day and night, the time-of-day light, a fixed light, and the
//! text size — each written to the profile's config and applied live through
//! the bus (`Change::Config` → `theme::set_settings`).
//!
//! **One difference, said on the screen**: a webview cannot read the
//! timezone's coordinates, so "Auto" and the time-of-day light follow a fixed
//! clock (day from 06:00 to 18:00) rather than the sun (`theme`'s module
//! docs). The language follows the device: the desktop app has no language
//! picker either.

use eidola_app_core::config::{
    AppearanceSetting, FONT_SCALE_DEFAULT, FONT_SCALE_MAX, FONT_SCALE_MIN, LightCharacter,
    TimeOfDayTint, font_scale_step_down, font_scale_step_up,
};
use eidola_app_core::error::AppError;
use gpui::{
    App, ClickEvent, Context, EventEmitter, InteractiveElement as _, IntoElement, MouseButton,
    ParentElement as _, Render, Role, ScrollHandle, SharedString, StatefulInteractiveElement as _,
    Styled as _, Window, div, prelude::FluentBuilder as _, px,
};
use gpui_component::{ActiveTheme as _, Icon};
use gpui_kit_assets::IconName;

use super::i18n::msg;
use super::nav::NavEvent;
use super::probe::Probe as _;
use super::stores::Stores;
use super::widgets::{error_note, header, keep_keyboard, note, section};
use super::{GUTTER, TOUCH};

pub struct AppearanceView {
    stores: Stores,
    scroll: ScrollHandle,
    error: Option<AppError>,
}

impl EventEmitter<NavEvent> for AppearanceView {}

impl AppearanceView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.account, |_, _, cx| cx.notify()).detach();
        Self {
            stores,
            scroll: ScrollHandle::new(),
            error: None,
        }
    }

    fn write(&mut self, result: Result<(), AppError>, cx: &mut Context<Self>) {
        self.error = result.err();
        cx.notify();
    }
}

/// One choice of a segmented row.
fn chip(
    name: String,
    label: SharedString,
    accessible: SharedString,
    selected: bool,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .id(SharedString::from(name.clone()))
        .probe(name, Role::RadioButton, accessible)
        .aria_selected(selected)
        .flex_1()
        .min_w_0()
        .min_h(TOUCH)
        .px(px(6.))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .text_size(px(15.))
        .when(selected, |d| {
            d.bg(theme.primary).text_color(theme.primary_foreground)
        })
        .when(!selected, |d| d.text_color(theme.foreground))
        .on_mouse_down(MouseButton::Left, keep_keyboard)
        .on_click(on_click)
        .child(label)
}

fn chips(children: Vec<gpui::AnyElement>, cx: &App) -> impl IntoElement {
    div()
        .mx(GUTTER)
        .p(px(3.))
        .flex()
        .gap_1()
        .rounded_full()
        .bg(cx.theme().secondary)
        .children(children)
}

impl Render for AppearanceView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let config = self.stores.account.read(cx).config().clone();
        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));

        let day_night: Vec<gpui::AnyElement> = [
            (
                AppearanceSetting::System,
                "system",
                msg::appearance_system(cx),
            ),
            (AppearanceSetting::Auto, "auto", msg::appearance_auto(cx)),
            (AppearanceSetting::Day, "day", msg::appearance_day(cx)),
            (AppearanceSetting::Night, "night", msg::appearance_night(cx)),
        ]
        .into_iter()
        .map(|(value, key, label)| {
            let accessible = msg::appearance_day_night_choice(cx, label.to_string());
            chip(
                format!("appearance/day-night/{key}"),
                label,
                accessible,
                config.appearance == value,
                cx.listener(move |this, _, _, cx| {
                    let r = this.stores.core.set_appearance(value);
                    this.write(r, cx)
                }),
                cx,
            )
            .into_any_element()
        })
        .collect();

        let tint: Vec<gpui::AnyElement> = [
            (TimeOfDayTint::On, "on", msg::appearance_on(cx)),
            (TimeOfDayTint::Off, "off", msg::appearance_off(cx)),
        ]
        .into_iter()
        .map(|(value, key, label)| {
            let accessible = msg::appearance_time_of_day_choice(cx, label.to_string());
            chip(
                format!("appearance/time-of-day/{key}"),
                label,
                accessible,
                config.time_of_day_tint == value,
                cx.listener(move |this, _, _, cx| {
                    let r = this.stores.core.set_time_of_day_tint(value);
                    this.write(r, cx)
                }),
                cx,
            )
            .into_any_element()
        })
        .collect();

        let light: Vec<gpui::AnyElement> = [
            (LightCharacter::Cool, "cool", msg::appearance_light_cool(cx)),
            (
                LightCharacter::Neutral,
                "neutral",
                msg::appearance_light_neutral(cx),
            ),
            (LightCharacter::Warm, "warm", msg::appearance_light_warm(cx)),
        ]
        .into_iter()
        .map(|(value, key, label)| {
            let accessible = msg::appearance_light_choice(cx, label.to_string());
            chip(
                format!("appearance/light/{key}"),
                label,
                accessible,
                config.light_character == value,
                cx.listener(move |this, _, _, cx| {
                    let r = this.stores.core.set_light_character(value);
                    this.write(r, cx)
                }),
                cx,
            )
            .into_any_element()
        })
        .collect();

        let scale = config.font_scale;
        let percent = format!("{}%", (scale * 100.0).round() as i32);
        let at_min = scale <= FONT_SCALE_MIN + 1e-3;
        let at_max = scale >= FONT_SCALE_MAX - 1e-3;
        let theme = cx.theme().clone();
        let step = |name: &'static str,
                    icon: IconName,
                    label: SharedString,
                    enabled: bool,
                    next: f32,
                    cx: &mut Context<Self>| {
            div()
                .id(name)
                .probe(name, Role::Button, label)
                .size(TOUCH)
                .flex()
                .items_center()
                .justify_center()
                .rounded_full()
                .bg(theme.secondary)
                .when(!enabled, |d| d.opacity(0.4))
                .on_mouse_down(MouseButton::Left, keep_keyboard)
                .when(enabled, |d| {
                    d.on_click(cx.listener(move |this, _, _, cx| {
                        let r = this.stores.core.set_font_scale(next);
                        this.write(r, cx)
                    }))
                })
                .child(Icon::new(icon).size(px(20.)))
        };
        let size_row = div()
            .mx(GUTTER)
            .flex()
            .items_center()
            .gap_3()
            .child(step(
                "appearance/text-size/smaller",
                IconName::Minus,
                msg::appearance_text_smaller(cx, percent.clone()),
                !at_min,
                font_scale_step_down(scale),
                cx,
            ))
            .child(
                div()
                    .id("appearance-text-size")
                    .probe("appearance/text-size", Role::Label, percent.clone())
                    .flex_1()
                    .text_center()
                    .font_family(eidola_theme::FONT_FAMILY)
                    .text_size(px(19.))
                    .child(percent.clone()),
            )
            .child(step(
                "appearance/text-size/larger",
                IconName::Plus,
                msg::appearance_text_larger(cx, percent.clone()),
                !at_max,
                font_scale_step_up(scale),
                cx,
            ))
            .child(
                div()
                    .id("appearance-text-size-reset")
                    .probe(
                        "appearance/text-size/reset",
                        Role::Button,
                        msg::appearance_text_reset_name(cx),
                    )
                    .min_h(TOUCH)
                    .px(px(12.))
                    .flex()
                    .items_center()
                    .text_size(px(15.))
                    .text_color(theme.link)
                    .on_mouse_down(MouseButton::Left, keep_keyboard)
                    .on_click(cx.listener(|this, _, _, cx| {
                        let r = this.stores.core.set_font_scale(FONT_SCALE_DEFAULT);
                        this.write(r, cx)
                    }))
                    .child(msg::appearance_text_reset(cx)),
            );

        let mut body = div()
            .id("appearance-body")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.scroll)
            .pb(px(32.))
            .child(section(msg::appearance_day_night(cx), cx))
            .child(chips(day_night, cx))
            .child(note(msg::appearance_day_night_note(cx), cx))
            .child(section(msg::appearance_time_of_day(cx), cx))
            .child(chips(tint, cx))
            .child(note(msg::appearance_time_of_day_note(cx), cx));
        if config.time_of_day_tint == TimeOfDayTint::Off {
            body = body
                .child(section(msg::appearance_light(cx), cx))
                .child(chips(light, cx))
                .child(note(msg::appearance_light_note(cx), cx));
        }
        body = body
            .child(section(msg::appearance_text_size(cx), cx))
            .child(size_row)
            .child(note(msg::appearance_text_size_note(cx), cx))
            .child(section(msg::appearance_language(cx), cx))
            .child(note(msg::appearance_language_note(cx), cx));
        if let Some(error) = &self.error {
            body = body.child(error_note(
                "appearance/error",
                msg::appearance_failed(cx, error.to_string()),
                cx,
            ));
        }
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "appearance",
                msg::appearance_title(cx),
                Some(back),
                None,
                cx,
            ))
            .child(body)
    }
}
