//! Circadian — the look Eidola's apps share: the palettes, the Newsreader
//! faces and the prose typography.
//!
//! Everything here is pure: palettes are built from settings, the schedule is
//! resolved from inputs a host supplies (its OS appearance, its wall clock, the
//! sun where it knows it), and the prose style is built from the installed
//! theme. Each host keeps its own wiring — where the settings come from, when
//! to re-apply, which windows to refresh — so the desktop app and the mobile
//! app read as one product without sharing any of their state machinery.
//!
//! **Two axes**, resolved together in [`resolve`]:
//!
//! - **Day/night** ([`AppearanceSetting`]): which palette family is active.
//!   `system` tracks the OS light/dark appearance; `day` / `night` pin one
//!   family; `auto` follows the sun — day is between sunrise and sunset, or the
//!   fixed 06:00–18:00 window without geography ([`DayPhases::Clock`]).
//! - **Time of day** ([`TimeOfDayTint`]): `on` takes on the [`LightCharacter`]
//!   of the light right now (cool around sunrise, neutral at solar noon and
//!   midnight, warm around sunset); `off` pins the character to the user's
//!   choice. [`canonical_hour`] warps wall time so the real sunrise lands at
//!   canonical 06:00 and sunset at 18:00, and the slot table reads unchanged.
//!
//! The tinted variants are **derived, not hand-authored**: every color of the
//! neutral palette is blended a few percent toward a per-family blue/ember
//! anchor ([`circadian_day`], [`circadian_night`]), so the six palettes cannot
//! drift apart as colors are tuned — except that the day variants tint the
//! near-white paper grounds at a reduced amount, keeping the sheet continuous
//! while the cast reads in the chrome and mid-tones.
//!
//! The two anchors are the product's "good paper at noon, reading lamp at
//! midnight": **Day `#ffffff`**, a neutral near-white paper whose only warmth is
//! a few accent foregrounds; **Night `#15191e`**, a cool blue-grey dark whose
//! surfaces follow the blue-grey ramp while text stays warm-grey and the brand
//! warm orange. The warm-on-cool tension is deliberate; don't "fix" it by
//! cooling the foregrounds.
//!
//! The body font is **Newsreader** (SIL OFL 1.1), six static 16pt instances
//! embedded here ([`load_fonts`]; license text at `assets/fonts/OFL.txt`):
//! statics rather than the variable fonts because gpui's macOS text system does
//! not apply variable-font weight axes, so every weight request would resolve
//! to Regular. All six report the typographic family [`FONT_FAMILY`].

use std::borrow::Cow;

use gpui::{App, SharedString};
use gpui_component::{ThemeConfig, ThemeConfigColors, ThemeMode};

// The palette constructors and the schedule speak these types; they live in
// app-core config because they are also persisted settings.
pub use eidola_app_core::config::{AppearanceSetting, LightCharacter, TimeOfDayTint};

pub mod prose;

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
/// backs the prose heading weight ([`prose::prose_style`]); without the
/// static face a 500 request falls back to Regular.
const NEWSREADER_REGULAR_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Regular.ttf");
const NEWSREADER_ITALIC_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Italic.ttf");
const NEWSREADER_MEDIUM_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Medium.ttf");
const NEWSREADER_SEMIBOLD_TTF: &[u8] =
    include_bytes!("../assets/fonts/Newsreader16pt-SemiBold.ttf");
const NEWSREADER_BOLD_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Bold.ttf");
const NEWSREADER_BOLD_ITALIC_TTF: &[u8] =
    include_bytes!("../assets/fonts/Newsreader16pt-BoldItalic.ttf");

/// Register the six Newsreader faces with the app's text system. Idempotent
/// at the gpui layer, so a context that installs twice (a test building
/// several `App`s) needs no guard. A failure is the caller's to report: the
/// prose then falls back to the system UI font.
pub fn load_fonts(cx: &App) -> gpui::Result<()> {
    cx.text_system().add_fonts(vec![
        Cow::Borrowed(NEWSREADER_REGULAR_TTF),
        Cow::Borrowed(NEWSREADER_ITALIC_TTF),
        Cow::Borrowed(NEWSREADER_MEDIUM_TTF),
        Cow::Borrowed(NEWSREADER_SEMIBOLD_TTF),
        Cow::Borrowed(NEWSREADER_BOLD_TTF),
        Cow::Borrowed(NEWSREADER_BOLD_ITALIC_TTF),
    ])
}

// ---------------------------------------------------------------------------
// The two axes
// ---------------------------------------------------------------------------

/// Both circadian settings, as resolved from config. A host holds them in
/// its own theme state and passes them to [`resolve`]. `character` is the
/// user's *fixed* choice, honored only while `tint` is `Off`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThemeSettings {
    pub appearance: AppearanceSetting,
    pub tint: TimeOfDayTint,
    pub character: LightCharacter,
    /// The base type-scale factor (`1.0` = designed sizes). Multiplies the
    /// theme's UI font size — which gpui-component installs as the window
    /// `rem_size`, so every `rems()`-relative size, line height, and padding
    /// rides it — and, through [`prose::prose_style`], the prose reading
    /// ramp. The single lever behind View → Actual Size / Zoom In / Zoom Out.
    pub font_scale: f32,
}

/// Where today's day and night actually fall — the geographic input to the
/// schedule. Derived from the system timezone's tzdb coordinates when
/// available (the desktop app's solar tables), otherwise the fixed clock
/// fallback.
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

pub fn circadian_day(character: LightCharacter, font_scale: f32) -> ThemeConfig {
    let name = match character {
        LightCharacter::Cool => "Circadian Sunrise",
        LightCharacter::Neutral => "Circadian Day",
        LightCharacter::Warm => "Circadian Sunset",
    };
    // `ThemeConfig` is `#[non_exhaustive]`: start from its default and
    // assign the fields this theme sets.
    let mut config = ThemeConfig::default();
    config.is_default = true;
    config.name = SharedString::new_static(name);
    config.mode = ThemeMode::Light;
    // Components (buttons, chrome, bylines) use the default system UI font.
    // Prose/narrative content opts into Newsreader explicitly via its own
    // `MarkdownStyle` (see `FONT_FAMILY`); the two font systems are kept
    // separate on purpose.
    config.font_family = None;
    // gpui-component installs this as the window `rem_size`, so scaling it
    // scales every `rems()`-relative chrome size, line height, and padding
    // in one move (see `prose::prose_style` for the prose ramp — the
    // one absolute-pixel exception that scales itself).
    config.font_size = Some(UI_FONT_SIZE * font_scale);
    config.mono_font_family = None;
    config.mono_font_size = Some(UI_FONT_SIZE * font_scale);
    config.radius = Some(6);
    config.radius_lg = Some(12);
    config.shadow = Some(true);
    config.colors = character_colors(day_colors(), false, character);
    config
}

/// The designed UI (chrome/component) font size at Actual Size — the value
/// gpui-component installs as the window `rem_size`. The type-scale factor
/// multiplies it.
pub const UI_FONT_SIZE: f32 = 14.;

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

pub fn circadian_night(character: LightCharacter, font_scale: f32) -> ThemeConfig {
    let name = match character {
        LightCharacter::Cool => "Circadian Dawn",
        LightCharacter::Neutral => "Circadian Night",
        LightCharacter::Warm => "Circadian Dusk",
    };
    // `ThemeConfig` is `#[non_exhaustive]`: start from its default and
    // assign the fields this theme sets.
    let mut config = ThemeConfig::default();
    config.is_default = true;
    config.name = SharedString::new_static(name);
    config.mode = ThemeMode::Dark;
    // System UI font for components; prose opts into Newsreader itself.
    // Scaled like Day (this size becomes the window `rem_size`).
    config.font_family = None;
    config.font_size = Some(UI_FONT_SIZE * font_scale);
    // Must be set explicitly: gpui-component only overwrites the live
    // `Theme.mono_font_size` when the config's is `Some` (theme
    // `schema.rs`), so leaving it `None` (via `..default`) would let dark
    // code blocks keep the framework default `px(13.)` — unscaled, and
    // stale-dependent on whether light was applied first.
    config.mono_font_family = None;
    config.mono_font_size = Some(UI_FONT_SIZE * font_scale);
    config.radius = Some(8);
    config.radius_lg = Some(12);
    config.shadow = Some(true);
    config.colors = character_colors(night_colors(), true, character);
    config
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
            font_scale: 1.0,
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
    fn font_scale_multiplies_the_ui_font_size() {
        // Actual Size is the designed UI size (the window rem_size) in both
        // families, for the body and mono ramps.
        let day = circadian_day(LightCharacter::Neutral, 1.0);
        assert_eq!(day.font_size, Some(UI_FONT_SIZE));
        assert_eq!(day.mono_font_size, Some(UI_FONT_SIZE));
        let night = circadian_night(LightCharacter::Neutral, 1.0);
        assert_eq!(night.font_size, Some(UI_FONT_SIZE));
        // Night MUST set `mono_font_size` explicitly (not `None`): gpui-component
        // only overwrites the live mono size when the config's is `Some`, so a
        // `None` here would strand dark code blocks at the framework default and
        // out of the zoom. Pinned in both families so the asymmetry can't return.
        assert_eq!(night.mono_font_size, Some(UI_FONT_SIZE));

        // A zoom multiplies both the body and mono ramps proportionally, in
        // both families — this is the single lever every rems()-relative chrome
        // size then rides.
        let day_big = circadian_day(LightCharacter::Neutral, 1.5);
        assert_eq!(day_big.font_size, Some(UI_FONT_SIZE * 1.5));
        assert_eq!(day_big.mono_font_size, Some(UI_FONT_SIZE * 1.5));
        let night_big = circadian_night(LightCharacter::Neutral, 2.0);
        assert_eq!(night_big.font_size, Some(UI_FONT_SIZE * 2.0));
        assert_eq!(night_big.mono_font_size, Some(UI_FONT_SIZE * 2.0));

        // The scale is orthogonal to the palette — colors are untouched.
        assert_eq!(day.colors.background, day_big.colors.background);
    }

    #[test]
    fn six_palettes_are_distinct() {
        let mut backgrounds = vec![];
        for ch in [
            LightCharacter::Cool,
            LightCharacter::Neutral,
            LightCharacter::Warm,
        ] {
            backgrounds.push(circadian_day(ch, 1.0).colors.background.clone().unwrap());
            backgrounds.push(circadian_night(ch, 1.0).colors.background.clone().unwrap());
        }
        let unique: std::collections::HashSet<_> = backgrounds.iter().collect();
        assert_eq!(
            unique.len(),
            6,
            "expected 6 distinct grounds: {backgrounds:?}"
        );
    }
}
