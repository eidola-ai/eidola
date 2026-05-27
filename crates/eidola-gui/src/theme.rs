//! Circadian — Eidola's theme.
//!
//! Two `ThemeConfig`s, "Circadian Day" (light) and "Circadian Night" (dark),
//! installed onto gpui-component's global `Theme`. Which one is active is
//! driven by the OS appearance: `Theme::sync_system_appearance` reads it
//! through `cx.window_appearance()` (or `window.appearance()` if a window is
//! passed) and applies the matching config. We then re-sync every open window
//! whenever the OS appearance changes.
//!
//! The starting palette is lifted from the marketing site
//! (`www.eidola.ai/index.html`). It will drift; treat the website as the
//! historical seed, not a contract.
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

    // Surfaces
    c.background = some("#faf7f2"); // bg
    c.foreground = some("#1e1c19"); // text
    c.border = some("#dcd6cc"); // rule
    c.input = some("#e8e2d8"); // card-border
    c.muted = some("#f4f0e9"); // code-bg
    c.muted_foreground = some("#696258"); // text-sub
    c.popover = some("#fffdf9"); // card
    c.popover_foreground = some("#1e1c19");
    c.accordion = some("#fffdf9");
    c.overlay = some("#1e1c1980");

    // Brand / interaction
    c.primary = some("#94522a"); // accent
    c.primary_foreground = some("#faf7f2"); // bg, reads best on the warm orange
    c.primary_hover = some("#824420"); // accent-text (slightly deeper)
    c.primary_active = some("#6e3818");
    c.ring = some("#94522a");
    c.caret = some("#94522a");
    c.selection = some("#94522a");
    c.link = some("#78411e"); // link
    c.link_hover = some("#94522a");

    // Subtle / chip surfaces — the website's tag-bg/tag-text
    c.secondary = some("#eee8de");
    c.secondary_foreground = some("#69553c");
    c.secondary_hover = some("#e6dfd0");
    c.secondary_active = some("#dcd3c0");
    c.accent = some("#eee8de");
    c.accent_foreground = some("#69553c");

    // Status — keep semantics distinct from the warm orange brand colour.
    c.danger = some("#b3401a");
    c.danger_foreground = some("#faf7f2");
    c.success = some("#3f7d4a");
    c.success_foreground = some("#faf7f2");
    c.warning = some("#a3741a");
    c.warning_foreground = some("#faf7f2");
    c.info = some("#3a6f8c");
    c.info_foreground = some("#faf7f2");

    // Chrome
    c.title_bar = some("#faf7f2");
    c.title_bar_border = some("#e8e2d8");
    c.tab_bar = some("#faf7f2");
    c.tab_bar_segmented = some("#eee8de");
    c.tab = some("#faf7f2");
    c.tab_active = some("#fffdf9");
    c.tab_active_foreground = some("#1e1c19");
    c.tab_foreground = some("#696258");
    c.sidebar = some("#f2eee6"); // footer-bg
    c.sidebar_border = some("#e8e2d8");
    c.sidebar_foreground = some("#1e1c19");
    c.sidebar_accent = some("#eee8de");
    c.sidebar_accent_foreground = some("#69553c");
    c.sidebar_primary = some("#94522a");
    c.sidebar_primary_foreground = some("#faf7f2");
    c.group_box = some("#f2eee6");
    c.group_box_foreground = some("#1e1c19");

    // Lists / scroll
    c.list = some("#faf7f2");
    c.list_even = some("#f4f0e9");
    c.list_head = some("#f2eee6");
    c.list_hover = some("#eee8de");
    c.scrollbar = some("#faf7f200");
    c.scrollbar_thumb = some("#dcd6cc");
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

    // Surfaces
    c.background = some("#141418");
    c.foreground = some("#d4d0c8");
    c.border = some("#302e34"); // rule
    c.input = some("#302e34"); // card-border
    c.muted = some("#1a1a20"); // code-bg
    c.muted_foreground = some("#827d73");
    c.popover = some("#1c1c21"); // card
    c.popover_foreground = some("#d4d0c8");
    c.accordion = some("#1c1c21");
    c.overlay = some("#000000a6");

    // Brand / interaction — softened orange on dark
    c.primary = some("#c39669");
    c.primary_foreground = some("#141418");
    c.primary_hover = some("#c89e73");
    c.primary_active = some("#a47d52");
    c.ring = some("#c39669");
    c.caret = some("#c39669");
    c.selection = some("#c39669");
    c.link = some("#c89e73");
    c.link_hover = some("#d4ae87");

    // Subtle / chip surfaces
    c.secondary = some("#28262c");
    c.secondary_foreground = some("#a09482");
    c.secondary_hover = some("#302e34");
    c.secondary_active = some("#3b393f");
    c.accent = some("#28262c");
    c.accent_foreground = some("#a09482");

    // Status
    c.danger = some("#d2664b");
    c.danger_foreground = some("#141418");
    c.success = some("#7eae8a");
    c.success_foreground = some("#141418");
    c.warning = some("#d2a45a");
    c.warning_foreground = some("#141418");
    c.info = some("#7fa4bf");
    c.info_foreground = some("#141418");

    // Chrome
    c.title_bar = some("#141418");
    c.title_bar_border = some("#302e34");
    c.tab_bar = some("#141418");
    c.tab_bar_segmented = some("#28262c");
    c.tab = some("#141418");
    c.tab_active = some("#1c1c21");
    c.tab_active_foreground = some("#d4d0c8");
    c.tab_foreground = some("#827d73");
    c.sidebar = some("#101014"); // footer-bg
    c.sidebar_border = some("#302e34");
    c.sidebar_foreground = some("#d4d0c8");
    c.sidebar_accent = some("#28262c");
    c.sidebar_accent_foreground = some("#a09482");
    c.sidebar_primary = some("#c39669");
    c.sidebar_primary_foreground = some("#141418");
    c.group_box = some("#1a1a20");
    c.group_box_foreground = some("#d4d0c8");

    // Lists / scroll
    c.list = some("#141418");
    c.list_even = some("#1a1a20");
    c.list_head = some("#1c1c21");
    c.list_hover = some("#28262c");
    c.scrollbar = some("#14141800");
    c.scrollbar_thumb = some("#302e34");
    c.scrollbar_thumb_hover = some("#4a474f");

    c
}

#[inline]
fn some(s: &'static str) -> Option<SharedString> {
    Some(SharedString::new_static(s))
}
