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
//! `assets/fonts/OFL.txt`.

use std::borrow::Cow;
use std::rc::Rc;
use std::time::Duration;

use eidola_app_core::config::{AppearanceSetting, TimeOfDayTint};
// Re-exported: the palette constructors and the driver's `theme` command
// speak this type; it lives in app-core config because it is also the
// persisted `light_character` setting.
pub use eidola_app_core::config::LightCharacter;
use gpui::{App, AsyncApp, Entity, Global, SharedString, Window};
use gpui_component::{Theme, ThemeConfig, ThemeConfigColors, ThemeMode};

use crate::stores::ConfigStore;

/// Prose font family. Must match the family name embedded in the bundled TTFs.
/// CoreText returns the typographic family (nid 16) when set, otherwise nid 1;
/// for our 16pt statics that resolves to `Newsreader 16pt` for every face.
/// Public so prose/narrative content can opt into Newsreader explicitly while
/// the theme leaves components on the system UI font.
pub const FONT_FAMILY: &str = "Newsreader 16pt";

/// 16pt static instances from `productiontype/Newsreader`. Six faces are the
/// minimum to make markdown bold/italic/heading weights render correctly:
/// `find_best_match` picks Medium for prose headings, SemiBold for the
/// editor's default heading ramp, Bold for **strong**, BoldItalic for
/// ***bold-italic***, Italic for `*emphasis*`, Regular for body. Medium (500)
/// backs the prose heading weight (`space_view::prose_style`); without the
/// static face a 500 request falls back to Regular.
const NEWSREADER_REGULAR_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Regular.ttf");
const NEWSREADER_ITALIC_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Italic.ttf");
const NEWSREADER_MEDIUM_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Medium.ttf");
const NEWSREADER_SEMIBOLD_TTF: &[u8] =
    include_bytes!("../assets/fonts/Newsreader16pt-SemiBold.ttf");
const NEWSREADER_BOLD_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Bold.ttf");
const NEWSREADER_BOLD_ITALIC_TTF: &[u8] =
    include_bytes!("../assets/fonts/Newsreader16pt-BoldItalic.ttf");

// ---------------------------------------------------------------------------
// The two axes
// ---------------------------------------------------------------------------

/// Both circadian settings, as resolved from config. Held in a [`Global`]
/// so [`apply`] can run from any context (config observer, clock task,
/// per-window appearance observer). `character` is the user's *fixed*
/// choice, honored only while `tint` is `Off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThemeSettings {
    pub appearance: AppearanceSetting,
    pub tint: TimeOfDayTint,
    pub character: LightCharacter,
}

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
}

impl Global for ThemeState {}

/// Where today's day and night actually fall — the geographic input to the
/// schedule. Derived from the system timezone's tzdb coordinates when
/// available ([`crate::solar`]), otherwise the fixed clock fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DayPhases {
    /// Sunrise/sunset in minutes since local midnight, from the timezone's
    /// representative coordinates.
    Solar { sunrise: f32, sunset: f32 },
    /// The sun never sets today: clock-slot characters; `auto` is day.
    PolarDay,
    /// The sun never rises today: clock-slot characters; `auto` is night.
    PolarNight,
    /// No geographic signal (fixed-offset zone, unreadable tzdb): the
    /// fixed clock schedule — sunrise 06:00, sunset 18:00.
    Clock,
}

/// Warp wall-clock time onto the canonical solar day: actual sunrise maps
/// to 06:00, sunset to 18:00, and the halves stretch/squeeze linearly, so
/// solar noon/midnight land at 12:00/00:00 and the slot table below reads
/// unchanged. Under [`DayPhases::Clock`] (and the polar cases, which have
/// no events to anchor to) this is the identity. Cyclic-safe: events that
/// straddle local midnight (pathological zones) still map continuously.
pub fn canonical_hour(now_min: f32, phases: DayPhases) -> f32 {
    let (sunrise, sunset) = match phases {
        DayPhases::Solar { sunrise, sunset } => (sunrise, sunset),
        _ => return (now_min / 60.0).rem_euclid(24.0),
    };
    let day_len = (sunset - sunrise).rem_euclid(1440.0);
    if day_len <= 0.0 || day_len >= 1440.0 {
        return (now_min / 60.0).rem_euclid(24.0);
    }
    let since_rise = (now_min - sunrise).rem_euclid(1440.0);
    if since_rise < day_len {
        6.0 + 12.0 * since_rise / day_len
    } else {
        let night_len = 1440.0 - day_len;
        let since_set = since_rise - day_len;
        (18.0 + 12.0 * since_set / night_len).rem_euclid(24.0)
    }
}

/// The light character of each canonical slot. The tinted windows are
/// brief, as the real transitions are — two canonical hours to each side
/// of the sun's events: dawn 04–06 + sunrise 06–08 are cool, sunset
/// 16–18 + dusk 18–20 are warm, and the long middles are neutral (day
/// 08–16, night 20–04). With solar [`DayPhases`] the canonical hour is
/// anchored to the real sun, so "sunrise 06–08" means the first sixth of
/// actual daylight.
pub fn character_for_hour(hour: f32) -> LightCharacter {
    let h = hour.rem_euclid(24.0);
    if (4.0..8.0).contains(&h) {
        LightCharacter::Cool
    } else if (8.0..16.0).contains(&h) {
        LightCharacter::Neutral
    } else if (16.0..20.0).contains(&h) {
        LightCharacter::Warm
    } else {
        LightCharacter::Neutral
    }
}

/// Resolve both axes to the (mode, character) to render. Pure — the
/// impure inputs (OS appearance, wall clock, solar events) are parameters.
pub fn resolve(
    settings: ThemeSettings,
    system_is_dark: bool,
    now_min: f32,
    phases: DayPhases,
) -> (ThemeMode, LightCharacter) {
    let canonical = canonical_hour(now_min, phases);
    let mode = match settings.appearance {
        AppearanceSetting::System => {
            if system_is_dark {
                ThemeMode::Dark
            } else {
                ThemeMode::Light
            }
        }
        AppearanceSetting::Day => ThemeMode::Light,
        AppearanceSetting::Night => ThemeMode::Dark,
        AppearanceSetting::Auto => match phases {
            DayPhases::PolarDay => ThemeMode::Light,
            DayPhases::PolarNight => ThemeMode::Dark,
            // Canonical 06–18 is exactly "between sunrise and sunset"
            // under Solar phases, and the fixed window under Clock.
            _ => {
                if (6.0..18.0).contains(&canonical) {
                    ThemeMode::Light
                } else {
                    ThemeMode::Dark
                }
            }
        },
    };
    let character = match settings.tint {
        TimeOfDayTint::On => character_for_hour(canonical),
        TimeOfDayTint::Off => settings.character,
    };
    (mode, character)
}

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

    if !cx.has_global::<ThemeState>() {
        cx.set_global(ThemeState {
            // Neutral defaults until `wire_config` reads the real config:
            // `system` matches the config default, `Off` + `Neutral` keeps
            // un-wired contexts (tests, the driver) on the anchor palettes.
            settings: ThemeSettings {
                appearance: AppearanceSetting::System,
                tint: TimeOfDayTint::Off,
                character: LightCharacter::Neutral,
            },
            applied: None,
            zone_cache: None,
        });
    }
    apply(None, cx);
}

/// Re-derive the palettes and mode from the current settings + clock + OS
/// appearance, install them on the global `Theme`, and refresh every open
/// window. The one path through which the theme ever changes.
pub fn apply(window: Option<&mut Window>, cx: &mut App) {
    let settings = cx.global::<ThemeState>().settings;
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

    {
        let theme = Theme::global_mut(cx);
        theme.light_theme = Rc::new(circadian_day(character));
        theme.dark_theme = Rc::new(circadian_night(character));
    }
    Theme::change(mode, window, cx);
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

/// Install the palettes for a specific (mode, character) pair, bypassing
/// the settings + clock resolution. QA seam for the UI driver's `theme`
/// command (and any test that wants to *see* a tinted palette) — production
/// always routes through [`apply`].
pub fn apply_fixed(mode: ThemeMode, character: LightCharacter, cx: &mut App) {
    {
        let theme = Theme::global_mut(cx);
        theme.light_theme = Rc::new(circadian_day(character));
        theme.dark_theme = Rc::new(circadian_night(character));
    }
    Theme::change(mode, None, cx);
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

    // The clock: once a minute, re-resolve and re-apply if the (mode,
    // character) pair moved to a new slot. App-lifetime task — the same
    // sanctioned `.detach()` as above.
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

/// Install new settings and re-apply if they changed.
fn set_settings(settings: ThemeSettings, cx: &mut App) {
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
    let result = cx.text_system().add_fonts(vec![
        Cow::Borrowed(NEWSREADER_REGULAR_TTF),
        Cow::Borrowed(NEWSREADER_ITALIC_TTF),
        Cow::Borrowed(NEWSREADER_MEDIUM_TTF),
        Cow::Borrowed(NEWSREADER_SEMIBOLD_TTF),
        Cow::Borrowed(NEWSREADER_BOLD_TTF),
        Cow::Borrowed(NEWSREADER_BOLD_ITALIC_TTF),
    ]);
    if let Err(e) = result {
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

// ---------------------------------------------------------------------------
// Tinting — the derived sunrise/sunset/dawn/dusk variants
// ---------------------------------------------------------------------------

/// The tint anchor + blend amount for a palette family and character.
/// `None` = the neutral anchor palette, untinted. The dark family blends
/// harder because a cast is less visible on dark grounds.
fn tint_spec(dark: bool, character: LightCharacter) -> Option<([u8; 3], f32)> {
    match (dark, character) {
        (_, LightCharacter::Neutral) => None,
        // Day: morning blue / low-sun ember over the neutral paper.
        (false, LightCharacter::Cool) => Some(([0x6d, 0x8f, 0xc0], 0.08)),
        (false, LightCharacter::Warm) => Some(([0xd0, 0x79, 0x3a], 0.08)),
        // Night: pre-dawn blue / dusk ember over the cool dark.
        (true, LightCharacter::Cool) => Some(([0x4a, 0x6f, 0xa5], 0.12)),
        (true, LightCharacter::Warm) => Some(([0xa8, 0x5c, 0x33], 0.12)),
    }
}

/// Blend one `#rrggbb` / `#rrggbbaa` value toward `target` by `amount`,
/// preserving any alpha suffix. Non-hex values pass through untouched.
fn tint_hex(hex: &str, target: [u8; 3], amount: f32) -> Option<String> {
    let raw = hex.strip_prefix('#')?;
    if raw.len() != 6 && raw.len() != 8 || !raw.is_ascii() {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(&raw[i..i + 2], 16).ok();
    let (r, g, b) = (byte(0)?, byte(2)?, byte(4)?);
    let blend =
        |c: u8, t: u8| (f32::from(c) * (1.0 - amount) + f32::from(t) * amount).round() as u8;
    Some(format!(
        "#{:02x}{:02x}{:02x}{}",
        blend(r, target[0]),
        blend(g, target[1]),
        blend(b, target[2]),
        &raw[6..]
    ))
}

/// Map every color field of a palette through `f(serde_field_name, hex)`;
/// `None` keeps the original. The serde round-trip covers new
/// `ThemeConfigColors` fields automatically instead of via a
/// hand-maintained field list.
fn map_colors(
    colors: ThemeConfigColors,
    f: impl Fn(&str, &str) -> Option<String>,
) -> ThemeConfigColors {
    let Ok(serde_json::Value::Object(map)) = serde_json::to_value(&colors) else {
        return colors;
    };
    let map: serde_json::Map<String, serde_json::Value> = map
        .into_iter()
        .map(|(k, v)| {
            let v = match v {
                serde_json::Value::String(s) => {
                    let mapped = f(&k, &s).unwrap_or_else(|| s.to_string());
                    serde_json::Value::String(mapped)
                }
                other => other,
            };
            (k, v)
        })
        .collect();
    serde_json::from_value(serde_json::Value::Object(map)).unwrap_or(colors)
}

/// Blend every color of a palette toward `target` by `amount` — one light
/// falling on the whole scene, so relative contrast is preserved.
fn tinted(colors: ThemeConfigColors, target: [u8; 3], amount: f32) -> ThemeConfigColors {
    map_colors(colors, |_, hex| tint_hex(hex, target, amount))
}

/// The day palette's paper grounds (serde field names): the near-white
/// page surfaces. In the day's tinted variants these take the cast at a
/// reduced amount ([`DAY_PAPER_TINT_FACTOR`]) so the sheet itself stays
/// close to neutral day's paper — the light's character reads in the
/// chrome and mid-tones more than in the page, and moving between the
/// three day palettes doesn't re-paper the whole window.
const DAY_PAPER_SOFT_TINT: &[&str] = &[
    "background",
    "popover.background",
    "accordion.background",
    "title_bar.background",
    "tab_bar.background",
    "tab.background",
    "tab.active.background",
    "list.background",
    "muted.background",
];

/// The fraction of the day tint the paper grounds receive.
const DAY_PAPER_TINT_FACTOR: f32 = 0.6;

fn character_colors(
    neutral: ThemeConfigColors,
    dark: bool,
    ch: LightCharacter,
) -> ThemeConfigColors {
    match tint_spec(dark, ch) {
        // Night tints uniformly; day softens the paper grounds.
        Some((target, amount)) if dark => tinted(neutral, target, amount),
        Some((target, amount)) => map_colors(neutral, |field, hex| {
            let amount = if DAY_PAPER_SOFT_TINT.contains(&field) {
                amount * DAY_PAPER_TINT_FACTOR
            } else {
                amount
            };
            tint_hex(hex, target, amount)
        }),
        None => neutral,
    }
}

// ---------------------------------------------------------------------------
// Day
// ---------------------------------------------------------------------------

fn circadian_day(character: LightCharacter) -> ThemeConfig {
    let name = match character {
        LightCharacter::Cool => "Circadian Sunrise",
        LightCharacter::Neutral => "Circadian Day",
        LightCharacter::Warm => "Circadian Sunset",
    };
    ThemeConfig {
        is_default: true,
        name: SharedString::new_static(name),
        mode: ThemeMode::Light,
        // Components (buttons, chrome, bylines) use the default system UI font.
        // Prose/narrative content opts into Newsreader explicitly via its own
        // `MarkdownStyle` (see `FONT_FAMILY`); the two font systems are kept
        // separate on purpose.
        font_family: None,
        font_size: Some(14.),
        mono_font_family: None,
        mono_font_size: Some(14.),
        radius: Some(6),
        radius_lg: Some(12),
        shadow: Some(true),
        colors: character_colors(day_colors(), false, character),
        ..ThemeConfig::default()
    }
}

fn day_colors() -> ThemeConfigColors {
    let mut c = ThemeConfigColors::default();

    // Surfaces — a neutral near-white paper. Cards/rules translate up in
    // lightness so they keep their relative depth on the bright ground; the
    // chip/chrome family below is neutral grey. The base carries no cast —
    // the sunrise/sunset tint is what gives day its cool/warm character.
    c.background = some("#ffffff"); // anchor: neutral paper
    c.foreground = some("#000000"); // ink
    c.border = some("#d8d8d8"); // hairline rule
    c.input = some("#e4e4e4"); // card-border
    c.muted = some("#fbfbfb"); // code-bg
    c.muted_foreground = some("#586169"); // text-sub
    c.popover = some("#fdfdfd"); // card
    c.popover_foreground = some("#191c1e");
    c.accordion = some("#fdfdfd");
    c.overlay = some("#191c1e80");

    // brand / interaction
    c.primary = some("#566169");
    c.primary_foreground = some("#fbfaf8");
    c.primary_hover = some("#415461");
    c.primary_active = some("#364350");
    c.ring = some("#4c6372");
    c.caret = some("#4c6372");
    c.selection = some("#4c6372");
    c.link = some("#1e5478");
    c.link_hover = some("#2a6a94");

    // Subtle / chip surfaces — neutral greys in the family of the separator
    // band (`muted`), with warm-brown foregrounds — day's mirror of night's
    // cool-grounds / warm-text tension. The base grounds and chrome stay
    // neutral; the day's cool/warm character arrives with the circadian tint.
    c.secondary = some("#ececec");
    c.secondary_foreground = some("#66615b");
    c.secondary_hover = some("#e3e3e3");
    c.secondary_active = some("#d7d7d7");
    c.accent = some("#ececec");
    c.accent_foreground = some("#615344");

    // Status — keep semantics distinct from the blue-grey brand colour.
    c.danger = some("#b3401a");
    c.danger_foreground = some("#fefaf5");
    c.success = some("#3f7d4a");
    c.success_foreground = some("#fefaf5");
    c.warning = some("#a3741a");
    c.warning_foreground = some("#fefaf5");
    c.info = some("#3a6f8c");
    c.info_foreground = some("#fefaf5");

    // Chrome
    // gpui-component Root's built-in Linux CSD frame would paint a 1px
    // `window_border` rectangle at the window surface's very edge (around
    // our transparent shadow margin). Our chrome (`chrome.rs`) draws the
    // real frame; pin this transparent so the vestigial one never shows.
    // Belt-and-braces: `chrome::themed_root` also disables Root's border
    // outright via `bordered(false)`.
    c.window_border = some("#00000000");
    c.title_bar = some("#fafafa");
    c.title_bar_border = some("#e4e4e4");
    c.tab_bar = some("#fafafa");
    c.tab_bar_segmented = some("#ececec");
    c.tab = some("#fafafa");
    c.tab_active = some("#fdfdfd");
    c.tab_active_foreground = some("#191c1e");
    c.tab_foreground = some("#586169");
    c.sidebar = some("#f1f1f1"); // neutral, like the chips
    c.sidebar_border = some("#e4e4e4");
    c.sidebar_foreground = some("#191c1e");
    c.sidebar_accent = some("#ececec");
    c.sidebar_accent_foreground = some("#3e454c");
    c.sidebar_primary = some("#566169");
    c.sidebar_primary_foreground = some("#fbfaf8");
    c.group_box = some("#f1f1f1");
    c.group_box_foreground = some("#191c1e");

    // lists / scroll
    c.list = some("#fafafa");
    c.list_even = some("#f4f4f4");
    c.list_head = some("#f1f1f1");
    c.list_hover = some("#ececec");
    c.scrollbar = some("#fafafa00");
    c.scrollbar_thumb = some("#d8d8d8");
    c.scrollbar_thumb_hover = some("#989898");
    c
}

// ---------------------------------------------------------------------------
// Night
// ---------------------------------------------------------------------------

fn circadian_night(character: LightCharacter) -> ThemeConfig {
    let name = match character {
        LightCharacter::Cool => "Circadian Dawn",
        LightCharacter::Neutral => "Circadian Night",
        LightCharacter::Warm => "Circadian Dusk",
    };
    ThemeConfig {
        is_default: true,
        name: SharedString::new_static(name),
        mode: ThemeMode::Dark,
        // System UI font for components; prose opts into Newsreader itself.
        font_family: None,
        font_size: Some(14.),
        radius: Some(8),
        radius_lg: Some(12),
        shadow: Some(true),
        colors: character_colors(night_colors(), true, character),
        ..ThemeConfig::default()
    }
}

fn night_colors() -> ThemeConfigColors {
    let mut c = ThemeConfigColors::default();

    // Surfaces — a blue-grey ramp derived from the anchor (#15191e keeps
    // R < G < B all the way up, so every elevated surface stays in the
    // anchor's cool family). The old palette's purple-grey neutrals
    // (#302e34 etc.) clashed with the new ground; everything here is
    // re-derived, not carried over.
    c.background = some("#15191e"); // anchor: cool blue-grey dark
    c.foreground = some("#ffffff"); // warm-grey — the reading-lamp tension
    c.border = some("#2c343d"); // rule
    c.input = some("#2c343d"); // card-border
    c.muted = some("#1b2026"); // code-bg, one step above the ground
    c.muted_foreground = some("#9e9e9e");
    c.popover = some("#20262d"); // card
    c.popover_foreground = some("#d4d0c8");
    c.accordion = some("#20262d");
    c.overlay = some("#000000a6");

    // Brand / interaction — softened warm orange on the cool dark
    c.primary = some("#c39669");
    c.primary_foreground = some("#15191e");
    c.primary_hover = some("#c89e73");
    c.primary_active = some("#a47d52");
    c.ring = some("#c39669");
    c.caret = some("#c39669");
    c.selection = some("#c39669");
    c.link = some("#c89e73");
    c.link_hover = some("#d4ae87");

    // Subtle / chip surfaces — cool grounds, warm foregrounds
    c.secondary = some("#262d35");
    c.secondary_foreground = some("#a89c88");
    c.secondary_hover = some("#2c343d");
    c.secondary_active = some("#333c46");
    c.accent = some("#262d35");
    c.accent_foreground = some("#a89c88");

    // Status
    c.danger = some("#d2664b");
    c.danger_foreground = some("#15191e");
    c.success = some("#7eae8a");
    c.success_foreground = some("#15191e");
    c.warning = some("#d2a45a");
    c.warning_foreground = some("#15191e");
    c.info = some("#7fa4bf");
    c.info_foreground = some("#15191e");

    // Chrome
    // Transparent for the same reason as Day: our chrome owns the frame.
    c.window_border = some("#00000000");
    c.title_bar = some("#15191e");
    c.title_bar_border = some("#2c343d");
    c.tab_bar = some("#15191e");
    c.tab_bar_segmented = some("#262d35");
    c.tab = some("#15191e");
    c.tab_active = some("#20262d");
    c.tab_active_foreground = some("#d4d0c8");
    c.tab_foreground = some("#9e9e9e");
    c.sidebar = some("#10141a"); // a step below the ground
    c.sidebar_border = some("#2c343d");
    c.sidebar_foreground = some("#d4d0c8");
    c.sidebar_accent = some("#262d35");
    c.sidebar_accent_foreground = some("#a89c88");
    c.sidebar_primary = some("#cfbdab");
    c.sidebar_primary_foreground = some("#15191e");
    c.group_box = some("#1b2026");
    c.group_box_foreground = some("#d4d0c8");

    // Lists / scroll
    c.list = some("#15191e");
    c.list_even = some("#1b2026");
    c.list_head = some("#20262d");
    c.list_hover = some("#262d35");
    c.scrollbar = some("#4e515500");
    c.scrollbar_thumb = some("#4d5156");
    c.scrollbar_thumb_hover = some("#5f6367");

    c
}

#[inline]
fn some(s: &'static str) -> Option<SharedString> {
    Some(SharedString::new_static(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_schedule_covers_the_six_slots() {
        // Dawn 04–06 and sunrise 06–08 are bluish (sunrise ±2h).
        for h in [4.0, 5.9, 6.0, 7.9] {
            assert_eq!(character_for_hour(h), LightCharacter::Cool, "hour {h}");
        }
        // The long neutral day, 08–16.
        for h in [8.0, 12.0, 15.9] {
            assert_eq!(character_for_hour(h), LightCharacter::Neutral, "hour {h}");
        }
        // Sunset 16–18 and dusk 18–20 are orange (sunset ±2h).
        for h in [16.0, 17.9, 18.0, 19.9] {
            assert_eq!(character_for_hour(h), LightCharacter::Warm, "hour {h}");
        }
        // The long neutral night, 20–04.
        for h in [20.0, 23.5, 0.0, 3.9] {
            assert_eq!(character_for_hour(h), LightCharacter::Neutral, "hour {h}");
        }
        // Out-of-range hours wrap instead of panicking.
        assert_eq!(character_for_hour(28.0), LightCharacter::Cool);
        assert_eq!(character_for_hour(-1.0), LightCharacter::Neutral);
    }

    #[test]
    fn canonical_hour_is_identity_without_geography() {
        for phases in [DayPhases::Clock, DayPhases::PolarDay, DayPhases::PolarNight] {
            for min in [0.0, 361.0, 720.0, 1439.0] {
                assert!(
                    (canonical_hour(min, phases) - min / 60.0).abs() < 1e-4,
                    "{phases:?} at {min}min"
                );
            }
        }
        // The fixed clock schedule *is* Solar{06:00, 18:00}.
        let clockish = DayPhases::Solar {
            sunrise: 360.0,
            sunset: 1080.0,
        };
        for min in [0.0, 200.0, 700.0, 1200.0] {
            assert!((canonical_hour(min, clockish) - min / 60.0).abs() < 1e-4);
        }
    }

    #[test]
    fn canonical_hour_warps_to_the_real_sun() {
        // A winter day at mid-latitude: sunrise 08:00, sunset 16:30.
        let phases = DayPhases::Solar {
            sunrise: 8.0 * 60.0,
            sunset: 16.5 * 60.0,
        };
        // Sunrise maps to canonical 06:00, sunset to 18:00.
        assert!((canonical_hour(8.0 * 60.0, phases) - 6.0).abs() < 1e-4);
        assert!((canonical_hour(16.5 * 60.0, phases) - 18.0).abs() < 1e-4);
        // Solar noon (midpoint of daylight, 12:15) maps to canonical 12:00.
        assert!((canonical_hour(12.25 * 60.0, phases) - 12.0).abs() < 1e-4);
        // 16:00 wall time is late daylight — canonical sunset slot (16–18),
        // so December's sunset character arrives before 16:30.
        let c = canonical_hour(16.0 * 60.0, phases);
        assert!(
            (16.0..18.0).contains(&c),
            "16:00 on a short day is canonical sunset, got {c}"
        );
        assert_eq!(character_for_hour(c), LightCharacter::Warm);
        // 20:00 is past dusk (18–20) on this short day — the long neutral
        // night has already begun.
        let c = canonical_hour(20.0 * 60.0, phases);
        assert!((20.0..24.0).contains(&c), "got {c}");
        assert_eq!(character_for_hour(c), LightCharacter::Neutral);
        // The night wraps continuously through midnight back to sunrise.
        let just_before_rise = canonical_hour(7.9 * 60.0, phases);
        assert!(
            (5.9..6.0).contains(&just_before_rise),
            "got {just_before_rise}"
        );
    }

    fn s(appearance: AppearanceSetting, tint: TimeOfDayTint) -> ThemeSettings {
        ThemeSettings {
            appearance,
            tint,
            character: LightCharacter::Neutral,
        }
    }

    #[test]
    fn resolve_maps_each_appearance_setting() {
        let off = |a| s(a, TimeOfDayTint::Off);
        let noon = 12.0 * 60.0;
        // System follows the OS flag.
        assert_eq!(
            resolve(
                off(AppearanceSetting::System),
                false,
                noon,
                DayPhases::Clock
            )
            .0,
            ThemeMode::Light
        );
        assert_eq!(
            resolve(off(AppearanceSetting::System), true, noon, DayPhases::Clock).0,
            ThemeMode::Dark
        );
        // Day/Night pin regardless of OS, clock, and sun.
        assert_eq!(
            resolve(
                off(AppearanceSetting::Day),
                true,
                23.0 * 60.0,
                DayPhases::PolarNight
            )
            .0,
            ThemeMode::Light
        );
        assert_eq!(
            resolve(
                off(AppearanceSetting::Night),
                false,
                noon,
                DayPhases::PolarDay
            )
            .0,
            ThemeMode::Dark
        );
        // Auto follows the clock fallback (day 06:00–18:00) without
        // geography, ignoring the OS flag.
        assert_eq!(
            resolve(
                off(AppearanceSetting::Auto),
                true,
                9.0 * 60.0,
                DayPhases::Clock
            )
            .0,
            ThemeMode::Light
        );
        assert_eq!(
            resolve(
                off(AppearanceSetting::Auto),
                false,
                5.0 * 60.0,
                DayPhases::Clock
            )
            .0,
            ThemeMode::Dark
        );
        assert_eq!(
            resolve(
                off(AppearanceSetting::Auto),
                false,
                18.0 * 60.0,
                DayPhases::Clock
            )
            .0,
            ThemeMode::Dark
        );
    }

    #[test]
    fn resolve_auto_follows_the_sun_when_it_has_one() {
        let auto = s(AppearanceSetting::Auto, TimeOfDayTint::Off);
        // Winter: sunrise 08:00, sunset 16:30 — 07:00 is still night and
        // 16:00 still day, both opposite the fixed 06–18 window's answer.
        let winter = DayPhases::Solar {
            sunrise: 8.0 * 60.0,
            sunset: 16.5 * 60.0,
        };
        assert_eq!(resolve(auto, false, 7.0 * 60.0, winter).0, ThemeMode::Dark);
        assert_eq!(
            resolve(auto, false, 16.0 * 60.0, winter).0,
            ThemeMode::Light
        );
        assert_eq!(resolve(auto, false, 17.0 * 60.0, winter).0, ThemeMode::Dark);
        // Polar days/nights force the mode outright.
        assert_eq!(
            resolve(auto, false, 12.0 * 60.0, DayPhases::PolarNight).0,
            ThemeMode::Dark
        );
        assert_eq!(
            resolve(auto, false, 0.0, DayPhases::PolarDay).0,
            ThemeMode::Light
        );
    }

    #[test]
    fn resolve_tint_axis_gates_the_character() {
        let on = s(AppearanceSetting::System, TimeOfDayTint::On);
        assert_eq!(
            resolve(on, false, 7.0 * 60.0, DayPhases::Clock).1,
            LightCharacter::Cool
        );
        assert_eq!(
            resolve(on, false, 16.0 * 60.0, DayPhases::Clock).1,
            LightCharacter::Warm
        );
        // On + geography: the character follows the warped (solar) hour —
        // 16:00 on a short winter day is already sunset-orange.
        let winter = DayPhases::Solar {
            sunrise: 8.0 * 60.0,
            sunset: 16.5 * 60.0,
        };
        assert_eq!(
            resolve(on, false, 16.0 * 60.0, winter).1,
            LightCharacter::Warm
        );
        // Off: the user's fixed character wins, whatever the sun is doing.
        let mut fixed = s(AppearanceSetting::System, TimeOfDayTint::Off);
        fixed.character = LightCharacter::Warm;
        assert_eq!(
            resolve(fixed, false, 7.0 * 60.0, winter).1,
            LightCharacter::Warm
        );
        fixed.character = LightCharacter::Cool;
        assert_eq!(
            resolve(fixed, false, 16.0 * 60.0, DayPhases::Clock).1,
            LightCharacter::Cool
        );
    }

    #[test]
    fn tint_hex_blends_and_preserves_alpha() {
        // 50% toward white from black is mid-grey.
        assert_eq!(
            tint_hex("#000000", [0xff, 0xff, 0xff], 0.5).as_deref(),
            Some("#808080")
        );
        // Zero amount is identity.
        assert_eq!(
            tint_hex("#15191e", [0x4a, 0x6f, 0xa5], 0.0).as_deref(),
            Some("#15191e")
        );
        // An alpha suffix survives untouched.
        assert_eq!(
            tint_hex("#1e1c1980", [0xff, 0xff, 0xff], 0.5).as_deref(),
            Some("#8f8e8c80")
        );
        // Non-hex values are left to the caller to pass through.
        assert_eq!(tint_hex("red", [0, 0, 0], 0.5), None);
        assert_eq!(tint_hex("#12", [0, 0, 0], 0.5), None);
    }

    #[test]
    fn tinted_palettes_shift_every_color_and_keep_the_untinted_shape() {
        let neutral = day_colors();
        let sunrise = character_colors(day_colors(), false, LightCharacter::Cool);
        // The anchor background moved toward blue…
        assert_ne!(neutral.background, sunrise.background);
        // …but no color was dropped or invented by the serde round-trip.
        let count = |c: &ThemeConfigColors| {
            let serde_json::Value::Object(map) = serde_json::to_value(c).unwrap() else {
                panic!("colors serialize to an object");
            };
            map.values().filter(|v| v.is_string()).count()
        };
        assert_eq!(count(&neutral), count(&sunrise));
        // Neutral is the identity — noon *is* the anchor palette.
        let noon = character_colors(day_colors(), false, LightCharacter::Neutral);
        assert_eq!(noon.background, neutral.background);
        assert_eq!(noon.overlay, neutral.overlay);
        // The overlay's alpha suffix survives tinting.
        let dusk = character_colors(night_colors(), true, LightCharacter::Warm);
        assert!(dusk.overlay.as_ref().unwrap().ends_with("a6"));
    }

    /// Channel spread (max − min) — a cheap saturation proxy.
    fn spread(hex: &Option<SharedString>) -> i32 {
        let raw = hex.as_ref().unwrap().strip_prefix('#').unwrap();
        let byte = |i: usize| i32::from_str_radix(&raw[i..i + 2], 16).unwrap();
        let (r, g, b) = (byte(0), byte(2), byte(4));
        r.max(g).max(b) - r.min(g).min(b)
    }

    #[test]
    fn day_chrome_is_neutral_grey_with_warm_foregrounds() {
        let day = day_colors();
        // The base is fully neutral — the paper ground and chrome carry no
        // cast (the sunrise/sunset tint is what colors them). Every ground
        // and chip surface is a true grey, in the separator band's (`muted`)
        // neutral family.
        for (field, hex) in [
            ("background", &day.background),
            ("title_bar", &day.title_bar),
            ("tab_bar", &day.tab_bar),
            ("secondary", &day.secondary),
            ("secondary_hover", &day.secondary_hover),
            ("secondary_active", &day.secondary_active),
            ("accent", &day.accent),
            ("sidebar", &day.sidebar),
            ("sidebar_accent", &day.sidebar_accent),
            ("tab_bar_segmented", &day.tab_bar_segmented),
            ("list_even", &day.list_even),
            ("list_head", &day.list_head),
            ("list_hover", &day.list_hover),
            ("group_box", &day.group_box),
        ] {
            assert_eq!(spread(hex), 0, "{field} must be a neutral grey: {hex:?}");
        }
        // …while a few accent foregrounds keep the warm-brown tension (the
        // mirror of night's cool grounds / warm text).
        assert!(spread(&day.secondary_foreground) > 0);
        assert!(spread(&day.sidebar_accent_foreground) > 0);
    }

    #[test]
    fn day_paper_grounds_take_a_softer_tint() {
        let target = [0x6d, 0x8f, 0xc0];
        let sunrise = character_colors(day_colors(), false, LightCharacter::Cool);
        // The page ground gets the reduced tint, not the full one…
        let soft = tint_hex("#ffffff", target, 0.08 * DAY_PAPER_TINT_FACTOR).unwrap();
        let full = tint_hex("#ffffff", target, 0.08).unwrap();
        assert_eq!(sunrise.background.as_deref(), Some(soft.as_str()));
        assert_ne!(sunrise.background.as_deref(), Some(full.as_str()));
        // …while chrome fields take the full tint.
        let full_chip = tint_hex("#ececec", target, 0.08).unwrap();
        assert_eq!(sunrise.secondary.as_deref(), Some(full_chip.as_str()));
        // Night tints uniformly — no soft list applies.
        let dawn = character_colors(night_colors(), true, LightCharacter::Cool);
        let uniform = tinted(night_colors(), [0x4a, 0x6f, 0xa5], 0.12);
        assert_eq!(dawn.background, uniform.background);
        assert_eq!(dawn.secondary, uniform.secondary);
    }

    #[test]
    fn six_palettes_are_distinct() {
        let mut backgrounds = vec![];
        for ch in [
            LightCharacter::Cool,
            LightCharacter::Neutral,
            LightCharacter::Warm,
        ] {
            backgrounds.push(circadian_day(ch).colors.background.clone().unwrap());
            backgrounds.push(circadian_night(ch).colors.background.clone().unwrap());
        }
        let unique: std::collections::HashSet<_> = backgrounds.iter().collect();
        assert_eq!(
            unique.len(),
            6,
            "expected 6 distinct grounds: {backgrounds:?}"
        );
    }
}
