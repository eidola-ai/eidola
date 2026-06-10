//! Circadian — Eidola's theme.
//!
//! Two `ThemeConfig`s, "Circadian Day" (light) and "Circadian Night" (dark),
//! installed onto gpui-component's global `Theme`. Which one is active is
//! driven by the OS appearance: `Theme::sync_system_appearance` reads it
//! through `cx.window_appearance()` (or `window.appearance()` if a window is
//! passed) and applies the matching config. We then re-sync every open window
//! whenever the OS appearance changes.
//!
//! The palettes are anchored on two fixed backgrounds chosen for the
//! product's "good paper at noon, reading lamp at midnight" feel:
//!
//! - **Day**: `#fefaf5` (254,250,245) — warm paper. Every other day surface
//!   is the same warm family, translated up to track the brighter ground.
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
//! binary. We ship five faces — Regular / Italic / SemiBold / Bold /
//! BoldItalic — because gpui's macOS text system does **not** apply
//! variable-font weight axes: each registered TTF is one face with the
//! properties of its default instance, and `font_kit::matching::find_best_match`
//! picks the closest face per weight request. With only a variable upright +
//! italic registered, every weight request resolved to Regular; with the
//! statics it resolves correctly (heading SEMIBOLD, **bold** BOLD, etc.).
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

use gpui::{App, SharedString, Window};
use gpui_component::{Theme, ThemeConfig, ThemeConfigColors, ThemeMode};

/// Body font family. Must match the family name embedded in the bundled TTFs.
/// CoreText returns the typographic family (nid 16) when set, otherwise nid 1;
/// for our 16pt statics that resolves to `Newsreader 16pt` for every face.
const FONT_FAMILY: &str = "Newsreader 16pt";

/// 16pt static instances from `productiontype/Newsreader`. Five faces are the
/// minimum to make markdown bold/italic/heading weights render correctly:
/// `find_best_match` picks SemiBold for h2-h5, Bold for h1 and **strong**,
/// BoldItalic for ***bold-italic***, Italic for `*emphasis*`, Regular for
/// body. Without a SemiBold the headings would still bold-fall-back; we ship
/// it for the visual cue between heading and body.
const NEWSREADER_REGULAR_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Regular.ttf");
const NEWSREADER_ITALIC_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Italic.ttf");
const NEWSREADER_SEMIBOLD_TTF: &[u8] =
    include_bytes!("../assets/fonts/Newsreader16pt-SemiBold.ttf");
const NEWSREADER_BOLD_TTF: &[u8] = include_bytes!("../assets/fonts/Newsreader16pt-Bold.ttf");
const NEWSREADER_BOLD_ITALIC_TTF: &[u8] =
    include_bytes!("../assets/fonts/Newsreader16pt-BoldItalic.ttf");

/// Install the Circadian themes onto the global `Theme` and apply whichever
/// matches the current OS appearance. Call once after `gpui_component::init`.
pub fn install(cx: &mut App) {
    load_fonts(cx);

    {
        let theme = Theme::global_mut(cx);
        theme.light_theme = Rc::new(circadian_day());
        theme.dark_theme = Rc::new(circadian_night());
    }
    Theme::sync_system_appearance(None, cx);
}

fn load_fonts(cx: &App) {
    // Idempotent at the gpui layer: re-adding the same family is a no-op
    // beyond a small bookkeeping cost, so tests that build multiple `App`s
    // (and therefore re-run `install`) don't need to guard.
    let result = cx.text_system().add_fonts(vec![
        Cow::Borrowed(NEWSREADER_REGULAR_TTF),
        Cow::Borrowed(NEWSREADER_ITALIC_TTF),
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
/// builder for each window we open.
pub fn observe_window_appearance(window: &mut Window) {
    window
        .observe_window_appearance(|window, cx| {
            Theme::sync_system_appearance(Some(window), cx);
        })
        .detach();
}

// ---------------------------------------------------------------------------
// Day
// ---------------------------------------------------------------------------

fn circadian_day() -> ThemeConfig {
    ThemeConfig {
        is_default: true,
        name: SharedString::new_static("Circadian Day"),
        mode: ThemeMode::Light,
        font_family: Some(SharedString::new_static(FONT_FAMILY)),
        radius: Some(8),
        radius_lg: Some(12),
        shadow: Some(true),
        colors: day_colors(),
        ..ThemeConfig::default()
    }
}

fn day_colors() -> ThemeConfigColors {
    let mut c = ThemeConfigColors::default();

    // Surfaces — every neutral is the anchor's warm family, translated up
    // in lightness so cards/chips/rules keep their relative depth on the
    // brighter paper.
    c.background = some("#fefaf5"); // anchor: warm paper
    c.foreground = some("#1e1c19"); // warm ink
    c.border = some("#e0d9cf"); // hairline rule
    c.input = some("#ece5db"); // card-border
    c.muted = some("#f8f3ec"); // code-bg
    c.muted_foreground = some("#696258"); // text-sub
    c.popover = some("#fffefb"); // card
    c.popover_foreground = some("#1e1c19");
    c.accordion = some("#fffefb");
    c.overlay = some("#1e1c1980");

    // Brand / interaction
    c.primary = some("#94522a"); // warm orange
    c.primary_foreground = some("#fefaf5"); // bg, reads best on the warm orange
    c.primary_hover = some("#824420"); // slightly deeper
    c.primary_active = some("#6e3818");
    c.ring = some("#94522a");
    c.caret = some("#94522a");
    c.selection = some("#94522a");
    c.link = some("#78411e");
    c.link_hover = some("#94522a");

    // Subtle / chip surfaces
    c.secondary = some("#f2ebe1");
    c.secondary_foreground = some("#69553c");
    c.secondary_hover = some("#eae2d3");
    c.secondary_active = some("#e0d6c3");
    c.accent = some("#f2ebe1");
    c.accent_foreground = some("#69553c");

    // Status — keep semantics distinct from the warm orange brand colour.
    c.danger = some("#b3401a");
    c.danger_foreground = some("#fefaf5");
    c.success = some("#3f7d4a");
    c.success_foreground = some("#fefaf5");
    c.warning = some("#a3741a");
    c.warning_foreground = some("#fefaf5");
    c.info = some("#3a6f8c");
    c.info_foreground = some("#fefaf5");

    // Chrome
    c.title_bar = some("#fefaf5");
    c.title_bar_border = some("#ece5db");
    c.tab_bar = some("#fefaf5");
    c.tab_bar_segmented = some("#f2ebe1");
    c.tab = some("#fefaf5");
    c.tab_active = some("#fffefb");
    c.tab_active_foreground = some("#1e1c19");
    c.tab_foreground = some("#696258");
    c.sidebar = some("#f6f1e9");
    c.sidebar_border = some("#ece5db");
    c.sidebar_foreground = some("#1e1c19");
    c.sidebar_accent = some("#f2ebe1");
    c.sidebar_accent_foreground = some("#69553c");
    c.sidebar_primary = some("#94522a");
    c.sidebar_primary_foreground = some("#fefaf5");
    c.group_box = some("#f6f1e9");
    c.group_box_foreground = some("#1e1c19");

    // Lists / scroll
    c.list = some("#fefaf5");
    c.list_even = some("#f8f3ec");
    c.list_head = some("#f6f1e9");
    c.list_hover = some("#f2ebe1");
    c.scrollbar = some("#fefaf500");
    c.scrollbar_thumb = some("#e0d9cf");
    c.scrollbar_thumb_hover = some("#a39a8a");

    c
}

// ---------------------------------------------------------------------------
// Night
// ---------------------------------------------------------------------------

fn circadian_night() -> ThemeConfig {
    ThemeConfig {
        is_default: true,
        name: SharedString::new_static("Circadian Night"),
        mode: ThemeMode::Dark,
        font_family: Some(SharedString::new_static(FONT_FAMILY)),
        radius: Some(8),
        radius_lg: Some(12),
        shadow: Some(true),
        colors: night_colors(),
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
    c.foreground = some("#d4d0c8"); // warm-grey — the reading-lamp tension
    c.border = some("#2c343d"); // rule
    c.input = some("#2c343d"); // card-border
    c.muted = some("#1b2026"); // code-bg, one step above the ground
    c.muted_foreground = some("#8a8478");
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
    c.tab_foreground = some("#8a8478");
    c.sidebar = some("#10141a"); // a step below the ground
    c.sidebar_border = some("#2c343d");
    c.sidebar_foreground = some("#d4d0c8");
    c.sidebar_accent = some("#262d35");
    c.sidebar_accent_foreground = some("#a89c88");
    c.sidebar_primary = some("#c39669");
    c.sidebar_primary_foreground = some("#15191e");
    c.group_box = some("#1b2026");
    c.group_box_foreground = some("#d4d0c8");

    // Lists / scroll
    c.list = some("#15191e");
    c.list_even = some("#1b2026");
    c.list_head = some("#20262d");
    c.list_hover = some("#262d35");
    c.scrollbar = some("#15191e00");
    c.scrollbar_thumb = some("#2c343d");
    c.scrollbar_thumb_hover = some("#46505c");

    c
}

#[inline]
fn some(s: &'static str) -> Option<SharedString> {
    Some(SharedString::new_static(s))
}
