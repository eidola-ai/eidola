//! Circadian — Eidola's theme.
//!
//! Two axes, resolved together in [`resolve`] and applied by [`apply`]:
//!
//! - **Day/night** ([`AppearanceSetting`], config key `appearance`): which
//!   palette family is active. `system` (default) tracks the OS light/dark
//!   appearance; `day` / `night` pin one family; `auto` follows the sun —
//!   day is between (timezone-approximated) sunrise and sunset, or the
//!   fixed 06:00–18:00 window without geography.
//! - **Time of day** ([`TimeOfDayTint`], config key `time_of_day_tint`):
//!   `on` follows the sun — the palette takes on the [`LightCharacter`] of
//!   the light right now: bluish around sunrise, neutral at solar
//!   noon/midnight, warm orange around sunset. `off` pins the character to
//!   the user's `light_character` config choice instead.
//!
//! **Geography from the timezone** ([`crate::solar`]): the IANA zone name
//! resolves to representative coordinates via the OS's own tzdb tables,
//! and the sunrise equation turns those into today's [`DayPhases`] — no
//! location permission, no network. [`canonical_hour`] then warps wall
//! time so actual sunrise lands at canonical 06:00 and sunset at 18:00,
//! and the fixed slot table reads unchanged on the warped clock. The
//! tinted slots hug the sun's events (±2 canonical hours — the transitions
//! are brief in real light): Dawn (04–06, night+cool), Sunrise (06–08,
//! day+cool), Day (08–16, long neutral), Sunset (16–18, day+warm),
//! Dusk (18–20, night+warm), Night (20–04, long neutral) — six palettes
//! anchored to the real solar day, so December's sunset character arrives
//! near 16:30 and June's near 20:30. Zones without coordinates (`UTC`,
//! `Etc/*`) and polar no-event days degrade to the fixed clock schedule
//! ([`DayPhases::Clock`]). The tinted variants are *derived*, not
//! hand-authored: every color of the neutral palette is blended a few
//! percent toward a per-family blue/ember anchor ([`tinted`]), so the
//! palettes can't drift apart as colors are tuned — except that the day
//! variants tint the near-white paper grounds at a reduced amount
//! ([`DAY_PAPER_SOFT_TINT`]), keeping the sheet continuous with neutral
//! day while the cast reads in the chrome and mid-tones.
//!
//! Wiring: [`install`] loads fonts and applies the *neutral* palettes (so
//! tests and the driver stay deterministic); the production `run()` calls
//! [`wire_config`], which reads the persisted settings from `ConfigStore`,
//! re-applies on every config change, and starts a once-a-minute clock task
//! that re-applies when the resolved (mode, character) pair crosses a slot
//! boundary. Per-window OS-appearance observers re-apply too (only `system`
//! mode actually follows them).
//!
//! The base palettes are **neutral** — the light has no cool/warm character
//! at solar noon or solar midnight. Character is applied *only* in the ±2h
//! windows around sunrise and sunset (see `tinted` / `tint_spec`), where the
//! palette shifts cool toward dawn and warm toward dusk. This is the theme's
//! bid at "rootedness" — the chrome tracks the character of real light
//! through the embodied day. The two anchors are fixed backgrounds chosen
//! for the product's "good paper at noon, reading lamp at midnight" feel:
//!
//! - **Day**: `#ffffff` — a neutral near-white paper. Warmth in the base
//!   lives in a few accent *foregrounds* (warm-brown chip labels), never in
//!   the page ground or the chrome; the ground only takes a cast under the
//!   sunrise/sunset tint.
//! - **Night**: `#15191e` (21,25,30) — a cool blue-grey dark. Night
//!   *surfaces* (cards, chips, borders, rows) follow the blue-grey ramp of
//!   the anchor, while *text* stays warm-grey and the brand stays warm
//!   orange — the warm-on-cool tension is deliberate (lamplight on a dark
//!   desk), so don't "fix" it by cooling the foregrounds.
//!
//! An earlier iteration seeded these palettes from the marketing site; the
//! site is no longer the reference — these anchors are.
//!
//! The body font is **Newsreader** (SIL OFL 1.1), shipped as the upstream
//! `productiontype/Newsreader` 16pt static instances and embedded into the
//! binary. We ship six faces — Regular / Italic / Medium / SemiBold / Bold /
//! BoldItalic — because gpui's macOS text system does **not** apply
//! variable-font weight axes: each registered TTF is one face with the
//! properties of its default instance, and `font_kit::matching::find_best_match`
//! picks the closest face per weight request. With only a variable upright +
//! italic registered, every weight request resolved to Regular; with the
//! statics it resolves correctly (prose headings MEDIUM, **bold** BOLD, etc.).
//! Medium (500) is the prose heading weight (see `space_view::prose_style`);
//! without the static face a 500 request would fall back to Regular.
//!
//! Family names: the 16pt statics report `Newsreader 16pt` as their
//! typographic family (nid 16 — the SemiBold needs nid 16 to override its
//! nid 1 = `Newsreader 16pt SemiBold`, which is the canonical workaround for
//! the OS/2 4-style-per-family limit on Windows). The variable TTFs from
//! `google/fonts` reported the family as `Newsreader` and were a different
//! family bucket; we no longer ship them. License text is at
//! `assets/fonts/OFL.txt` in `eidola-theme`, which embeds the faces.

use std::rc::Rc;
use std::time::Duration;

use eidola_app_core::config::{AppearanceSetting, TimeOfDayTint};
// Re-exported: the palette constructors and the driver's `theme` command
// speak this type; it lives in app-core config because it is also the
// persisted `light_character` setting.
pub use eidola_app_core::config::LightCharacter;
use gpui::{App, AsyncApp, Entity, Global, SharedString, Window};
use gpui_component::{Theme, ThemeMode};

use crate::stores::ConfigStore;

// The shared look (palettes, faces, prose) lives in `eidola-theme`; this
// module wires it to the desktop app's config, clock, sun and windows.
pub use eidola_theme::{
    DayPhases, FONT_FAMILY, ThemeSettings, canonical_hour, character_for_hour, resolve,
};
use eidola_theme::{circadian_day, circadian_night};

/// Theme state global: the current settings, the (mode, character) pair
/// last applied — the clock task compares against it to detect slot
/// boundaries without re-deriving palettes every minute — and the
/// per-zone-name memo of the tzdb coordinate lookup (the only part of the
/// solar computation worth caching: it scans a file; the zone name re-check
/// each tick is a cheap readlink, so travel is picked up automatically).
struct ThemeState {
    settings: ThemeSettings,
    applied: Option<(ThemeMode, LightCharacter)>,
    zone_cache: Option<(String, Option<(f64, f64)>)>,
    /// The native window appearance last forced on the app ([`sync_chrome`]),
    /// `None` while the windows follow the OS.
    chrome: Option<gpui::WindowAppearance>,
}

impl Global for ThemeState {}

/// A snapshot of local civil time, from `libc::localtime_r` (no chrono
/// dependency; the circadian schedule is about the user's local day,
/// unlike the Record's UTC timestamps).
struct LocalNow {
    /// Minutes since local midnight.
    minutes: f32,
    /// Unix seconds — the solar equation's time input.
    unix: i64,
    /// The current UTC offset, for mapping solar events into local time.
    utc_offset_secs: i32,
}

fn local_now() -> LocalNow {
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let secs = unix as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // Safety: `secs` and `tm` are valid for the duration of the call.
    unsafe { libc::localtime_r(&secs, &mut tm) };
    LocalNow {
        minutes: (tm.tm_hour * 60 + tm.tm_min) as f32 + tm.tm_sec as f32 / 60.0,
        unix,
        utc_offset_secs: tm.tm_gmtoff as i32,
    }
}

/// Today's [`DayPhases`] from the system timezone, memoizing the tzdb
/// coordinate lookup per zone name in [`ThemeState`].
fn current_phases(now: &LocalNow, cx: &mut App) -> DayPhases {
    let Some(zone) = crate::solar::system_zone_name() else {
        return DayPhases::Clock;
    };
    let state = cx.global_mut::<ThemeState>();
    let coords = match &state.zone_cache {
        Some((cached_zone, coords)) if *cached_zone == zone => *coords,
        _ => {
            let coords = crate::solar::zone_coordinates(&zone);
            state.zone_cache = Some((zone, coords));
            coords
        }
    };
    let Some((lat, lon)) = coords else {
        return DayPhases::Clock;
    };
    match crate::solar::solar_events(lat, lon, now.unix, now.utc_offset_secs) {
        crate::solar::SolarEvents::Normal { sunrise, sunset } => {
            DayPhases::Solar { sunrise, sunset }
        }
        crate::solar::SolarEvents::PolarDay => DayPhases::PolarDay,
        crate::solar::SolarEvents::PolarNight => DayPhases::PolarNight,
    }
}

// ---------------------------------------------------------------------------
// Install & apply
// ---------------------------------------------------------------------------

/// Install the Circadian themes onto the global `Theme` and apply the
/// *neutral* palettes under the system appearance. Call once after
/// `gpui_component::init`. Tests and the driver stop here, so their
/// palettes are deterministic (never wall-clock-tinted); the production
/// `run()` follows up with [`wire_config`], which loads the persisted
/// settings and turns the clock on.
pub fn install(cx: &mut App) {
    load_fonts(cx);

    // Every post body — a model's reply included, while it streams — is a
    // `MarkdownEditor`, so `![alt](url)` in text the reader did not write
    // would otherwise choose a fetch: a request to any host (the app has no
    // HTTP client today, so that one fails, but only by accident) or an
    // unbounded read of any local path. Only the app's own assets resolve;
    // everything else shows its markdown. Set here because every context —
    // the app, the tests, the driver — runs `install`. A host that renders
    // files the reader opened (the notepad) sets `Any` after this call.
    cx.set_global(gpui_markdown_editor::image::ImageSources::EmbeddedOnly);

    if !cx.has_global::<ThemeState>() {
        cx.set_global(ThemeState {
            // Neutral defaults until `wire_config` reads the real config:
            // `system` matches the config default, `Off` + `Neutral` keeps
            // un-wired contexts (tests, the driver) on the anchor palettes.
            settings: ThemeSettings {
                appearance: AppearanceSetting::System,
                tint: TimeOfDayTint::Off,
                character: LightCharacter::Neutral,
                font_scale: eidola_app_core::config::FONT_SCALE_DEFAULT,
            },
            applied: None,
            zone_cache: None,
            chrome: None,
        });
    }
    apply(None, cx);
}

/// Re-derive the palettes and mode from the current settings + clock + OS
/// appearance, install them on the global `Theme`, and refresh every open
/// window. The one path through which the theme ever changes.
pub fn apply(window: Option<&mut Window>, cx: &mut App) {
    let settings = cx.global::<ThemeState>().settings;
    // Under `system` the OS appearance is an input, so any override has to be
    // gone before it is read — a forced appearance is what `effectiveAppearance`
    // would otherwise report back.
    if settings.appearance == AppearanceSetting::System {
        sync_chrome(None, cx);
    }
    let system_is_dark = ThemeMode::from(
        window
            .as_ref()
            .map(|w| w.appearance())
            .unwrap_or_else(|| cx.window_appearance()),
    )
    .is_dark();
    let now = local_now();
    let phases =
        if settings.tint == TimeOfDayTint::On || settings.appearance == AppearanceSetting::Auto {
            current_phases(&now, cx)
        } else {
            // Nothing reads the sun: skip the zone/tzdb work entirely.
            DayPhases::Clock
        };
    let (mode, character) = resolve(settings, system_is_dark, now.minutes, phases);
    if settings.appearance != AppearanceSetting::System {
        sync_chrome(Some(mode), cx);
    }

    {
        let theme = Theme::global_mut(cx);
        theme.light_theme = Rc::new(circadian_day(character, settings.font_scale));
        theme.dark_theme = Rc::new(circadian_night(character, settings.font_scale));
    }
    Theme::change(mode, window, cx);
    snapshot_ring_colors(cx);
    if cx.has_global::<ThemeState>() {
        cx.global_mut::<ThemeState>().applied = Some((mode, character));
    }

    // `Theme::change` refreshed the passed window (if any); reach the rest.
    // Re-entrant update on the window we were dispatched inside fails
    // harmlessly (it was already refreshed above).
    for handle in cx.windows() {
        handle.update(cx, |_, window, _| window.refresh()).ok();
    }
}

/// Make the native window chrome — the traffic lights, the window border —
/// match the palette on screen. AppKit draws them from the app's
/// `NSAppearance`, so a Night palette under a light system (or Day under a
/// dark one) left light-mode traffic lights on a dark page.
///
/// `None` clears the override and lets the windows follow the OS, which is
/// what `system` wants and what keeps its live Light/Dark tracking working: a
/// forced appearance stops windows reporting system changes, and is what the
/// OS appearance would read back as. Every other setting ignores the OS, so it
/// can force the resolved mode. Only a change is sent, so re-applying on the
/// clock does not keep re-setting the app's appearance.
fn sync_chrome(mode: Option<ThemeMode>, cx: &mut App) {
    let wanted = mode.map(|m| {
        if m.is_dark() {
            gpui::WindowAppearance::Dark
        } else {
            gpui::WindowAppearance::Light
        }
    });
    if cx.global::<ThemeState>().chrome != wanted {
        cx.global_mut::<ThemeState>().chrome = wanted;
        cx.set_window_appearance(wanted);
    }
}

/// Test seam: the native window appearance currently forced on the app, or
/// `None` while windows follow the OS. The platform half is a no-op under the
/// test platform, so this is what can be asserted.
#[doc(hidden)]
pub fn chrome_override_for_test(cx: &App) -> Option<gpui::WindowAppearance> {
    cx.global::<ThemeState>().chrome
}

/// Snapshot the focus ring's two colors out of the palette that was just
/// installed, so `probe_inner` — which has no `App` to ask — can build the
/// ring. Called from the two paths that ever change the theme. The ring is the
/// palette's own accent (`ring`: a muted blue-grey by day, the brand's warm
/// orange by night) at the day/night alpha; the spacer is the paper it sits on.
fn snapshot_ring_colors(cx: &App) {
    let theme = gpui_component::Theme::global(cx);
    let alpha = if theme.is_dark() {
        crate::focus::RING_ALPHA_NIGHT
    } else {
        crate::focus::RING_ALPHA_DAY
    };
    crate::focus::set_ring_colors(crate::focus::RingColors {
        ring: gpui::Hsla {
            a: alpha,
            ..theme.ring
        },
        paper: theme.background,
    });
}

/// The current base type-scale factor from the theme global (`1.0` when the
/// theme hasn't been installed yet, e.g. a bare test `App`). Read by the prose
/// surface ([`crate::space_view::prose_style`]) — the one place the type ramp
/// is spelled in absolute pixels rather than `rems()`, so it doesn't ride the
/// scaled `rem_size` automatically and must multiply the factor itself.
pub fn font_scale(cx: &App) -> f32 {
    if cx.has_global::<ThemeState>() {
        cx.global::<ThemeState>().settings.font_scale
    } else {
        eidola_app_core::config::FONT_SCALE_DEFAULT
    }
}

/// Install the palettes for a specific (mode, character) pair, bypassing
/// the settings + clock resolution. QA seam for the UI driver's `theme`
/// command (and any test that wants to *see* a tinted palette) — production
/// always routes through [`apply`].
pub fn apply_fixed(mode: ThemeMode, character: LightCharacter, cx: &mut App) {
    // Honor the current type scale so the driver's forced-palette previews and
    // any test that wants to *see* a tinted palette also reflect a zoom.
    let scale = font_scale(cx);
    {
        let theme = Theme::global_mut(cx);
        theme.light_theme = Rc::new(circadian_day(character, scale));
        theme.dark_theme = Rc::new(circadian_night(character, scale));
    }
    Theme::change(mode, None, cx);
    if cx.has_global::<ThemeState>() {
        sync_chrome(Some(mode), cx);
    }
    snapshot_ring_colors(cx);
    for handle in cx.windows() {
        handle.update(cx, |_, window, _| window.refresh()).ok();
    }
}

/// Point the theme at the persisted settings: seed from the `ConfigStore`
/// snapshot, re-apply whenever the config changes, and start the clock
/// task that advances the palette across slot boundaries. Called once from
/// `run()`; tests that never call this stay on the neutral install state.
pub fn wire_config(config: &Entity<ConfigStore>, cx: &mut App) {
    let read_settings = |store: &ConfigStore| {
        store.state().map(|s| ThemeSettings {
            appearance: s.appearance,
            tint: s.time_of_day_tint,
            character: s.light_character,
            font_scale: s.font_scale,
        })
    };

    if let Some(settings) = read_settings(config.read(cx)) {
        set_settings(settings, cx);
    }

    // App-lifetime observation — nothing to cancel it against, so
    // `.detach()` is sanctioned here (same rationale as the bus bridge).
    cx.observe(config, move |config, cx| {
        if let Some(settings) = read_settings(config.read(cx)) {
            set_settings(settings, cx);
        }
    })
    .detach();

    start_clock(cx);
}

/// Start the clock task that advances the palette across slot boundaries:
/// once a minute, re-resolve and re-apply if the (mode, character) pair moved
/// to a new slot. Called once per process — by [`wire_config`], or by a host
/// that drives the theme through [`set_settings`] instead of a `ConfigStore`.
/// App-lifetime task — the same sanctioned `.detach()` as the config observer.
pub fn start_clock(cx: &mut App) {
    cx.spawn(async move |cx: &mut AsyncApp| {
        loop {
            cx.background_executor()
                .timer(Duration::from_secs(60))
                .await;
            // The task dies with the executor at shutdown; `update` itself
            // is infallible at our pin (same idiom as the bus bridge).
            cx.update(reapply_if_slot_changed);
        }
    })
    .detach();
}

/// The settings the theme is currently resolved from.
pub fn settings(cx: &App) -> ThemeSettings {
    cx.global::<ThemeState>().settings
}

/// Install new settings and re-apply if they changed. The production app
/// reaches this through [`wire_config`]; a host without a `ConfigStore` calls
/// it directly (after [`install`]).
pub fn set_settings(settings: ThemeSettings, cx: &mut App) {
    let state = cx.global_mut::<ThemeState>();
    if state.settings != settings {
        state.settings = settings;
        apply(None, cx);
    }
}

/// The clock tick: re-apply only when the resolved (mode, character) pair
/// differs from what's on screen.
fn reapply_if_slot_changed(cx: &mut App) {
    let state = cx.global::<ThemeState>();
    let settings = state.settings;
    let applied = state.applied;
    let system_is_dark = ThemeMode::from(cx.window_appearance()).is_dark();
    let now = local_now();
    let phases = current_phases(&now, cx);
    if applied != Some(resolve(settings, system_is_dark, now.minutes, phases)) {
        apply(None, cx);
    }
}

fn load_fonts(cx: &App) {
    // Idempotent at the gpui layer: re-adding the same family is a no-op
    // beyond a small bookkeeping cost, so tests that build multiple `App`s
    // (and therefore re-run `install`) don't need to guard.
    if let Err(e) = eidola_theme::load_fonts(cx) {
        // Don't panic the app over a font failure — fall back to the system
        // UI font (which `ThemeConfig::font_family = None` resolves to).
        eprintln!("eidola-gui: failed to register Newsreader fonts: {e}");
    }
}

/// Subscribe a window to OS appearance changes so Light/Dark switches at the
/// system level are picked up live. Call from inside the `cx.open_window`
/// builder for each window we open. Routes through [`apply`], so a pinned
/// `day`/`night`/`auto` appearance correctly ignores the OS flip.
///
/// On Linux/Wayland the theme flip is spawned onto the foreground executor
/// rather than applied inline: gpui delivers appearance changes from inside
/// the xdg-desktop-portal event callback, which holds the Wayland client's
/// `RefCell` mutably (`gpui_linux` `wayland/client.rs`, `XDPEvent::
/// WindowAppearance`); applying the theme there would mark windows dirty and
/// the surrounding `update`'s effect flush would repaint synchronously, and
/// the glyph paint path (`is_subpixel_rendering_supported`) re-borrows that
/// same `RefCell` — "RefCell already mutably borrowed", a startup panic on
/// GNOME, which pushes an initial color-scheme event on every launch. A
/// spawned task runs on the next executor tick, outside the portal dispatch.
/// (`cx.defer` is NOT sufficient: deferred callbacks run in the same effect
/// flush, still inside the borrow.) Upstream fix would be gpui dropping the
/// client borrow before invoking window callbacks; until then this stays.
/// macOS has no such reentrancy, so it applies inline (no one-tick lag on the
/// Light/Dark flip).
pub fn observe_window_appearance(window: &mut Window) {
    window
        .observe_window_appearance(|window, cx| {
            #[cfg(target_os = "macos")]
            apply(Some(window), cx);

            #[cfg(not(target_os = "macos"))]
            {
                let handle = window.window_handle();
                cx.spawn(async move |cx| {
                    handle
                        .update(cx, |_, window, cx| {
                            apply(Some(window), cx);
                        })
                        .ok();
                })
                .detach();
            }
        })
        .detach();
}

/// Vertical padding for a single-line gpui-component `Input`, in rems so it
/// follows the type scale. The widget's default (8px a side inside a 2rem
/// frame) leaves less room than its 1.25rem line at our font sizes, and the
/// line is centered and clipped to that room, which cuts off descenders. A
/// quarter rem a side keeps the frame height and gives the line its full
/// height. Applied at every `Input` site.
pub const INPUT_PY: gpui::Rems = gpui::Rems(0.25);

/// The label of a default-size (Medium) gpui-component `Button`, at the 1rem
/// the app's buttons are designed at. The component sets a Medium button's
/// content to 0.875rem, which a `text_base()` on the button cannot reach, so
/// the label goes in as its own child instead of through `Button::label`.
pub fn button_label(label: impl Into<SharedString>) -> impl gpui::IntoElement {
    use gpui::{ParentElement as _, Styled as _};
    gpui::div()
        .min_w_0()
        .whitespace_nowrap()
        .text_ellipsis()
        .text_base()
        .child(label.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every post body is a `MarkdownEditor` rendering text the reader did
    /// not write, so the theme every context installs must leave only the
    /// app's own assets resolvable as images.
    #[gpui::test]
    fn install_admits_only_the_apps_own_images(cx: &mut gpui::TestAppContext) {
        use gpui_markdown_editor::image::ImageSources;
        cx.update(|cx| {
            gpui_component::init(cx);
            install(cx);
            assert_eq!(*cx.global::<ImageSources>(), ImageSources::EmbeddedOnly);
        });
    }
}
