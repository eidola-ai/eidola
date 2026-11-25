//! The Circadian look on the phone: the palettes, faces and prose the desktop
//! app uses (`eidola-theme`), wired to this app's config, its webview's
//! appearance (`prefers-color-scheme`, which follows the device) and its clock.
//!
//! One difference from the desktop app, stated rather than hidden: the
//! desktop app finds the sun from the timezone's tzdb coordinates, which a
//! webview cannot read, so here the schedule runs on the fixed clock
//! ([`DayPhases::Clock`]: sunrise 06:00, sunset 18:00) — the same degradation
//! the desktop app takes in a zone without coordinates.
//!
//! And one more: **the faces are the bundle's own**. gpui's web text system
//! shapes only fonts loaded into it (the browser's system fonts are reached
//! only by the canvas fallback, for emoji and CJK), so the desktop's system UI
//! font and its Menlo/Courier code faces do not exist here, and naming one
//! that does not resolve is a panic. The chrome is therefore set in the prose
//! face, Newsreader ([`UI_FAMILY`]), and code in KaTeX Typewriter
//! ([`MONO_FAMILY`]), which the markdown editor already embeds for math.

use std::rc::Rc;
use std::time::Duration;

use eidola_theme::{
    AppearanceSetting, DayPhases, LightCharacter, ThemeSettings, TimeOfDayTint, circadian_day,
    circadian_night, resolve,
};
use gpui::{App, Global, Window};
use gpui_component::{Theme, ThemeMode};
use gpui_markdown_editor::MarkdownStyle;

/// The chrome's face: the bundle's Newsreader (see the module docs).
pub const UI_FAMILY: &str = eidola_theme::FONT_FAMILY;
/// The code face: KaTeX Typewriter, embedded with the editor's math fonts.
pub const MONO_FAMILY: &str = "KaTeX_Typewriter";

struct ThemeState {
    settings: ThemeSettings,
    applied: Option<(ThemeMode, LightCharacter)>,
}

impl Global for ThemeState {}

/// Load the faces, refuse remote images in post bodies, and apply the
/// neutral palettes under the webview's appearance. After
/// `gpui_component::init`.
pub fn install(cx: &mut App) {
    if let Err(e) = eidola_theme::load_fonts(cx) {
        log::error!("failed to register Newsreader fonts: {e}");
    }
    if let Err(e) = gpui_markdown_editor::math::register_katex_fonts(cx.text_system()) {
        log::error!("failed to register the KaTeX fonts: {e}");
    }
    // Every post body renders text the reader did not write: only the app's
    // own assets resolve as images, never a URL in a model's answer.
    cx.set_global(gpui_markdown_editor::image::ImageSources::EmbeddedOnly);
    if !cx.has_global::<ThemeState>() {
        cx.set_global(ThemeState {
            settings: ThemeSettings {
                appearance: AppearanceSetting::System,
                tint: TimeOfDayTint::Off,
                character: LightCharacter::Neutral,
                font_scale: eidola_app_core::config::FONT_SCALE_DEFAULT,
            },
            applied: None,
        });
    }
    apply(None, cx);
}

/// The settings from the profile's config, applied if they changed.
pub fn set_settings(settings: ThemeSettings, cx: &mut App) {
    let state = cx.global_mut::<ThemeState>();
    if state.settings != settings {
        state.settings = settings;
        apply(None, cx);
    }
}

/// The settings a config snapshot carries.
pub fn settings_of(state: &eidola_app_core::ConfigState) -> ThemeSettings {
    ThemeSettings {
        appearance: state.appearance,
        tint: state.time_of_day_tint,
        character: state.light_character,
        font_scale: state.font_scale,
    }
}

/// Resolve the settings against the webview's appearance and the clock and
/// install the palettes. The one path the theme changes through.
pub fn apply(window: Option<&mut Window>, cx: &mut App) {
    let settings = cx.global::<ThemeState>().settings;
    let system_is_dark = ThemeMode::from(
        window
            .as_ref()
            .map(|w| w.appearance())
            .unwrap_or_else(|| cx.window_appearance()),
    )
    .is_dark();
    let (mode, character) = resolve(settings, system_is_dark, local_minutes(), DayPhases::Clock);
    {
        let theme = Theme::global_mut(cx);
        theme.light_theme = Rc::new(circadian_day(character, settings.font_scale));
        theme.dark_theme = Rc::new(circadian_night(character, settings.font_scale));
    }
    Theme::change(mode, window, cx);
    use_bundled_faces(cx);
    cx.global_mut::<ThemeState>().applied = Some((mode, character));
    for handle in cx.windows() {
        handle.update(cx, |_, window, _| window.refresh()).ok();
    }
}

/// Install a fixed (mode, character) pair, bypassing the settings and clock:
/// the development bundle's palette switch for screenshots.
#[cfg(feature = "dev")]
pub fn apply_fixed(mode: ThemeMode, cx: &mut App) {
    let scale = font_scale(cx);
    {
        let theme = Theme::global_mut(cx);
        theme.light_theme = Rc::new(circadian_day(LightCharacter::Neutral, scale));
        theme.dark_theme = Rc::new(circadian_night(LightCharacter::Neutral, scale));
    }
    Theme::change(mode, None, cx);
    use_bundled_faces(cx);
    for handle in cx.windows() {
        handle.update(cx, |_, window, _| window.refresh()).ok();
    }
}

/// Point the theme's UI and mono families at faces the bundle carries.
fn use_bundled_faces(cx: &mut App) {
    let theme = Theme::global_mut(cx);
    theme.font_family = UI_FAMILY.into();
    theme.mono_font_family = MONO_FAMILY.into();
}

/// Once a minute, re-apply when the resolved pair crossed a slot boundary.
/// App-lifetime, so detached: nothing ever cancels it.
pub fn start_clock(cx: &mut App) {
    cx.spawn(async move |cx| {
        loop {
            cx.background_executor()
                .timer(Duration::from_secs(60))
                .await;
            cx.update(|cx| {
                let state = cx.global::<ThemeState>();
                let settings = state.settings;
                let applied = state.applied;
                let dark = ThemeMode::from(cx.window_appearance()).is_dark();
                if applied != Some(resolve(settings, dark, local_minutes(), DayPhases::Clock)) {
                    apply(None, cx);
                }
            });
        }
    })
    .detach();
}

/// Follow the device's light/dark switch live (it reaches the webview as
/// `prefers-color-scheme`). Only `system` acts on it; [`apply`] decides.
pub fn observe_window_appearance(window: &mut Window) {
    window
        .observe_window_appearance(|window, cx| apply(Some(window), cx))
        .detach();
}

/// The type-scale factor in force.
pub fn font_scale(cx: &App) -> f32 {
    cx.try_global::<ThemeState>()
        .map(|s| s.settings.font_scale)
        .unwrap_or(eidola_app_core::config::FONT_SCALE_DEFAULT)
}

/// The prose style every post body and the composer share — the desktop
/// app's, at this app's type scale, so a post reads as what was typed.
///
/// **One difference, for the phone's width**: a gentler heading ramp (h1 1.5×,
/// h2 1.3×, h3 1.15×, h4–h6 1.05× the body, against the desktop's 2.5× / 1.75×
/// / 1.25× / 1.125×), so a heading does not take the screen from the words
/// beneath it. Size still carries the hierarchy, at the desktop's uniform
/// Medium weight.
pub fn prose_style(cx: &App) -> MarkdownStyle {
    let mut style =
        eidola_theme::prose::prose_style(cx, font_scale(cx)).heading_font_size(|level, base| {
            match level {
                1 => base * 1.5,
                2 => base * 1.3,
                3 => base * 1.15,
                _ => base * 1.05,
            }
        });
    style.mono_font_family = MONO_FAMILY.into();
    style.inline_code_font_family = MONO_FAMILY.into();
    style
}

/// Minutes since local midnight: the browser's clock on wasm32; natively
/// (tests) UTC, which only has to be a time of day.
fn local_minutes() -> f32 {
    #[cfg(target_family = "wasm")]
    {
        let now = js_sys::Date::new_0();
        (now.get_hours() * 60 + now.get_minutes()) as f32
    }
    #[cfg(not(target_family = "wasm"))]
    {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        ((secs % 86_400) / 60) as f32
    }
}
