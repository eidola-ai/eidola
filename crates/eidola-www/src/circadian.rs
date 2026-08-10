//! The circadian palette system, ported from the GUI's `theme.rs`.
//!
//! The website adopts the app's visual system verbatim: two hand-authored
//! *neutral* anchor palettes (day = "good paper at noon", night = "reading
//! lamp at midnight") from which the four tinted variants are *derived* by
//! blending every color toward a single light target — "one light falling
//! on the whole scene," so relative contrast is preserved.
//!
//! The constants here are a projection of
//! `crates/eidola-gui/src/theme.rs` (`day_colors` / `night_colors` /
//! `tint_spec` / `DAY_PAPER_TINT_FACTOR`) onto the smaller set of slots a
//! website needs. If the app palette changes, this module must be updated
//! to match; the unit tests pin the derived values so drift is loud.
//!
//! Output is a stylesheet defining CSS custom properties: the day-neutral
//! set on `:root` (with the night-neutral set behind
//! `prefers-color-scheme: dark` as the no-JS fallback), plus all six
//! palettes keyed by `:root[data-palette="..."]`, which `circadian.js`
//! selects at runtime from the sun's schedule.

/// A palette slot: name (becomes `--<name>`) and day/night neutral values.
///
/// Values are `#rrggbb` or `#rrggbbaa` (alpha preserved through tinting,
/// exactly like the app's `tint_hex`).
const SLOTS: &[(&str, &str, &str)] = &[
    // slot, day-neutral, night-neutral — from theme.rs day_colors()/night_colors()
    ("background", "#ffffff", "#15191e"),
    ("foreground", "#000000", "#ffffff"),
    ("border", "#d8d8d8", "#2c343d"),
    ("muted", "#fbfbfb", "#1b2026"),
    ("muted-foreground", "#586169", "#9e9e9e"),
    ("surface", "#fdfdfd", "#20262d"),
    ("surface-foreground", "#191c1e", "#d4d0c8"),
    ("primary", "#566169", "#c39669"),
    ("link", "#1e5478", "#c89e73"),
    ("link-hover", "#2a6a94", "#d4ae87"),
    ("selection", "#4c6372", "#c39669"),
    ("secondary", "#ececec", "#262d35"),
    ("secondary-foreground", "#66615b", "#a89c88"),
    ("accent-foreground", "#615344", "#a89c88"),
    ("scrollbar-thumb", "#d8d8d8", "#4d5156"),
    // Status — theme.rs's semantic colors; the site uses them for the
    // GFM alert blockquotes (site.css `.markdown-alert-*`).
    ("info", "#3a6f8c", "#7fa4bf"),
    ("success", "#3f7d4a", "#7eae8a"),
    ("warning", "#a3741a", "#d2a45a"),
    ("danger", "#b3401a", "#d2664b"),
];

/// Day slots that take only `DAY_PAPER_TINT_FACTOR` of the tint, so the
/// paper sheet stays continuous with neutral day while chrome and
/// mid-tones carry the cast (theme.rs `DAY_PAPER_SOFT_TINT`). Night tints
/// uniformly.
const DAY_PAPER_SOFT_TINT: &[&str] = &["background", "muted", "surface"];
const DAY_PAPER_TINT_FACTOR: f64 = 0.6;

/// Tint targets and amounts (theme.rs `tint_spec`). The dark family blends
/// harder because a cast is less visible on dark grounds.
const DAY_COOL: (u8, u8, u8) = (0x6d, 0x8f, 0xc0); // morning blue
const DAY_WARM: (u8, u8, u8) = (0xd0, 0x79, 0x3a); // low-sun ember
const NIGHT_COOL: (u8, u8, u8) = (0x4a, 0x6f, 0xa5); // pre-dawn blue
const NIGHT_WARM: (u8, u8, u8) = (0xa8, 0x5c, 0x33); // dusk ember
const DAY_AMOUNT: f64 = 0.08;
const NIGHT_AMOUNT: f64 = 0.12;

/// Blend a `#rrggbb`/`#rrggbbaa` color toward a target by `amount`,
/// preserving any alpha suffix — the app's `tint_hex` formula:
/// `out = round(c * (1 - amount) + target * amount)` per channel.
fn tint_hex(hex: &str, target: (u8, u8, u8), amount: f64) -> String {
    let raw = hex.trim_start_matches('#');
    let (rgb, alpha) = raw.split_at(6);
    let channel = |i: usize, t: u8| -> u8 {
        let c = u8::from_str_radix(&rgb[i..i + 2], 16).expect("valid hex palette constant");
        (f64::from(c) * (1.0 - amount) + f64::from(t) * amount).round() as u8
    };
    format!(
        "#{:02x}{:02x}{:02x}{}",
        channel(0, target.0),
        channel(2, target.1),
        channel(4, target.2),
        alpha
    )
}

/// The six palette keys, in the order the stylesheet emits them.
const PALETTES: &[&str] = &[
    "day-cool",
    "day-neutral",
    "day-warm",
    "night-cool",
    "night-neutral",
    "night-warm",
];

/// Resolve one slot's value in a given palette.
fn slot_value(slot: &str, day: &str, night: &str, palette: &str) -> String {
    let (base, is_day) = if palette.starts_with("day") {
        (day, true)
    } else {
        (night, false)
    };
    let target = match palette {
        "day-cool" => Some(DAY_COOL),
        "day-warm" => Some(DAY_WARM),
        "night-cool" => Some(NIGHT_COOL),
        "night-warm" => Some(NIGHT_WARM),
        _ => None,
    };
    match target {
        None => base.to_string(),
        Some(t) => {
            let mut amount = if is_day { DAY_AMOUNT } else { NIGHT_AMOUNT };
            if is_day && DAY_PAPER_SOFT_TINT.contains(&slot) {
                amount *= DAY_PAPER_TINT_FACTOR;
            }
            tint_hex(base, t, amount)
        }
    }
}

fn variables(palette: &str) -> String {
    let mut vars: String = SLOTS
        .iter()
        .map(|(slot, day, night)| {
            format!("  --{}: {};\n", slot, slot_value(slot, day, night, palette))
        })
        .collect();
    // Corner radius follows the family (theme.rs: day radius 6, night 8).
    let radius = if palette.starts_with("day") { 6 } else { 8 };
    vars.push_str(&format!("  --radius: {radius}px;\n"));
    vars
}

/// Emit the full palette stylesheet.
pub fn stylesheet() -> String {
    let mut css = String::from(
        "/* Generated by eidola-www (src/circadian.rs) — do not edit.\n\
         * The circadian palettes, ported from crates/eidola-gui/src/theme.rs.\n\
         * No-JS fallback: day-neutral, or night-neutral under\n\
         * prefers-color-scheme: dark. circadian.js picks the live palette. */\n",
    );
    css.push_str(&format!(":root {{\n{}}}\n", variables("day-neutral")));
    css.push_str(&format!(
        "@media (prefers-color-scheme: dark) {{\n:root {{\n{}}}\n}}\n",
        variables("night-neutral")
    ));
    for palette in PALETTES {
        css.push_str(&format!(
            ":root[data-palette=\"{}\"] {{\n{}}}\n",
            palette,
            variables(palette)
        ));
    }
    css
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tint_blends_toward_target() {
        // Day-cool chrome: blend(#ececec, #6d8fc0, 0.08) — full amount.
        assert_eq!(tint_hex("#ececec", DAY_COOL, 0.08), "#e2e5e8");
        // Day-cool paper: blend(#ffffff, #6d8fc0, 0.08 * 0.6) — softened.
        assert_eq!(tint_hex("#ffffff", DAY_COOL, 0.08 * 0.6), "#f8fafc");
        // Night-warm ground: blend(#15191e, #a85c33, 0.12) — uniform night.
        assert_eq!(tint_hex("#15191e", NIGHT_WARM, 0.12), "#272121");
    }

    #[test]
    fn tint_preserves_alpha_suffix() {
        assert_eq!(tint_hex("#191c1e80", DAY_COOL, 0.08), "#20252b80");
    }

    #[test]
    fn neutral_is_identity() {
        assert_eq!(
            slot_value("background", "#ffffff", "#15191e", "day-neutral"),
            "#ffffff"
        );
        assert_eq!(
            slot_value("background", "#ffffff", "#15191e", "night-neutral"),
            "#15191e"
        );
    }

    #[test]
    fn day_paper_slots_take_soft_tint() {
        // background is on the soft list; secondary is not.
        assert_eq!(
            slot_value("background", "#ffffff", "#15191e", "day-warm"),
            tint_hex("#ffffff", DAY_WARM, DAY_AMOUNT * DAY_PAPER_TINT_FACTOR)
        );
        assert_eq!(
            slot_value("secondary", "#ececec", "#262d35", "day-warm"),
            tint_hex("#ececec", DAY_WARM, DAY_AMOUNT)
        );
    }

    #[test]
    fn stylesheet_contains_all_palettes_and_fallbacks() {
        let css = stylesheet();
        for palette in PALETTES {
            assert!(css.contains(&format!(":root[data-palette=\"{palette}\"]")));
        }
        assert!(css.contains("prefers-color-scheme: dark"));
        assert!(css.contains("--background: #ffffff;"));
        assert!(css.contains("--background: #15191e;"));
    }

    #[test]
    fn status_slots_match_the_app() {
        // Pinned to theme.rs's Status blocks (day_colors/night_colors) so
        // drift from the app palette is loud, like every other slot.
        let css = stylesheet();
        for pinned in [
            "--info: #3a6f8c;",
            "--info: #7fa4bf;",
            "--success: #3f7d4a;",
            "--success: #7eae8a;",
            "--warning: #a3741a;",
            "--warning: #d2a45a;",
            "--danger: #b3401a;",
            "--danger: #d2664b;",
        ] {
            assert!(css.contains(pinned), "missing {pinned}");
        }
    }

    #[test]
    fn day_grounds_are_true_neutrals() {
        // Mirrors the app's tested invariant: every day ground/chip surface
        // is a true grey (R = G = B); character enters only via tinting.
        for slot in [
            "background",
            "border",
            "muted",
            "secondary",
            "scrollbar-thumb",
        ] {
            let (_, day, _) = SLOTS.iter().find(|(s, _, _)| *s == slot).unwrap();
            let raw = day.trim_start_matches('#');
            assert_eq!(raw[0..2], raw[2..4], "{slot} not neutral");
            assert_eq!(raw[2..4], raw[4..6], "{slot} not neutral");
        }
    }
}
