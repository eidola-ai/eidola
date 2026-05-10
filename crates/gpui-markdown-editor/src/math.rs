//! LaTeX math typesetting via RaTeX.
//!
//! Parse a LaTeX expression, lay it out, and walk RaTeX's
//! [`DisplayList`](ratex_types::display_item::DisplayList) emitting
//! native gpui paint primitives — `paint_quad` for fraction bars and
//! filled rectangles, `paint_path` for radical signs, and shaped glyph
//! runs for letters and operators. Math composes with the rest of the
//! editor's text and respects Day/Night themes; the only inputs that
//! depend on the cursor / theme are the color and font size at paint
//! time, so a single laid-out [`MathLayout`] can be repainted for
//! either theme without re-running the typesetter.
//!
//! # Hosting requirements
//!
//! Math glyphs come from 19 bundled KaTeX TTF faces. Hosting
//! applications must call [`register_katex_fonts`] with the gpui
//! text system at app init (or before the first math paint). The
//! call is idempotent.
//!
//! # Coordinate system
//!
//! RaTeX produces all coordinates in **em units** (1em = 1 multiple of
//! `font_size_px`). x increases right, y increases down, the origin is
//! the top-left of the bounding box, and the baseline lives at
//! `y = list.height`. We multiply em values by the caller's `em_px`
//! at paint time to project to absolute pixels.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::OnceLock;

use gpui::{
    App, Bounds, FontStyle, FontWeight, Hsla, Path, Pixels, Point, SharedString, Size, TextRun,
    TextSystem, Window, fill, point, px, size,
};
use ratex_layout::{LayoutOptions, layout, to_display_list};
use ratex_parser::parser::parse;
use ratex_types::display_item::{DisplayItem, DisplayList};
use ratex_types::math_style::MathStyle;
use ratex_types::path_command::PathCommand;

/// KaTeX TTFs to load. Mirrors `ratex-render`'s list, minus the
/// system-Unicode/CJK/emoji fallbacks — math content uses just these
/// 19 faces, and we'd rather not pull a few hundred KB of CJK fallback
/// when this crate is dropped into another gpui app.
const KATEX_FONT_FILES: &[&str] = &[
    "KaTeX_Main-Regular.ttf",
    "KaTeX_Main-Bold.ttf",
    "KaTeX_Main-Italic.ttf",
    "KaTeX_Main-BoldItalic.ttf",
    "KaTeX_Math-Italic.ttf",
    "KaTeX_Math-BoldItalic.ttf",
    "KaTeX_AMS-Regular.ttf",
    "KaTeX_Caligraphic-Regular.ttf",
    "KaTeX_Caligraphic-Bold.ttf",
    "KaTeX_Fraktur-Regular.ttf",
    "KaTeX_Fraktur-Bold.ttf",
    "KaTeX_SansSerif-Regular.ttf",
    "KaTeX_SansSerif-Bold.ttf",
    "KaTeX_SansSerif-Italic.ttf",
    "KaTeX_Script-Regular.ttf",
    "KaTeX_Typewriter-Regular.ttf",
    "KaTeX_Size1-Regular.ttf",
    "KaTeX_Size2-Regular.ttf",
    "KaTeX_Size3-Regular.ttf",
    "KaTeX_Size4-Regular.ttf",
];

static KATEX_FONTS_REGISTERED: OnceLock<()> = OnceLock::new();

/// Register the bundled KaTeX TTFs with `text_system`. Idempotent —
/// subsequent calls are no-ops. Hosts may call this at app init
/// alongside their own font loads, or rely on the editor to call it
/// lazily the first time math typesets.
pub fn register_katex_fonts(text_system: &Arc<TextSystem>) -> Result<(), String> {
    if KATEX_FONTS_REGISTERED.get().is_some() {
        return Ok(());
    }
    let mut fonts: Vec<Cow<'static, [u8]>> = Vec::with_capacity(KATEX_FONT_FILES.len());
    for filename in KATEX_FONT_FILES {
        let bytes = ratex_katex_fonts::ttf_bytes(filename)
            .ok_or_else(|| format!("KaTeX font {filename} missing from ratex-katex-fonts"))?;
        fonts.push(bytes);
    }
    text_system
        .add_fonts(fonts)
        .map_err(|e| format!("registering KaTeX fonts: {e}"))?;
    let _ = KATEX_FONTS_REGISTERED.set(());
    Ok(())
}

/// Initial math style for the typesetter. `Display` matches `$$...$$`
/// (large fractions, limits above/below operators); `Inline` matches
/// `$...$` (smaller, limits to the right).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathMode {
    Display,
    Inline,
}

/// A typeset math expression. Holds RaTeX's flat
/// [`DisplayList`](ratex_types::display_item::DisplayList); cheap to
/// keep around and repaint at different sizes / colors.
pub struct MathLayout {
    list: DisplayList,
}

/// Parse + lay out a LaTeX math expression. Returns the layout, ready
/// for repeated repainting. Pass `MathMode::Display` for `$$...$$`
/// constructs and `MathMode::Inline` for `$...$`.
pub fn typeset(latex: &str, mode: MathMode) -> Result<MathLayout, String> {
    let nodes = parse(latex).map_err(|e| e.to_string())?;
    let style = match mode {
        MathMode::Display => MathStyle::Display,
        MathMode::Inline => MathStyle::Text,
    };
    let options = LayoutOptions {
        style,
        ..LayoutOptions::default()
    };
    let lbox = layout(&nodes, &options);
    let list = to_display_list(&lbox);
    Ok(MathLayout { list })
}

impl MathLayout {
    /// Pixel dimensions at the given em-pixel size. `height` is the
    /// total visual height (above + below baseline).
    pub fn size(&self, em_px: Pixels) -> Size<Pixels> {
        let em = f32::from(em_px);
        Size {
            width: px(self.list.width as f32 * em),
            height: px((self.list.height + self.list.depth) as f32 * em),
        }
    }

    /// Vertical distance from the top of the layout to the math
    /// baseline. Needed for inline math so the surrounding text and
    /// the math share a baseline.
    pub fn baseline(&self, em_px: Pixels) -> Pixels {
        px(self.list.height as f32 * f32::from(em_px))
    }

    /// Walk the DisplayList and emit native gpui paint commands.
    /// `origin` is the top-left of the math region in absolute paint
    /// coordinates. `color` is applied to every item whose layout
    /// color is the default (BLACK) — so dark/light themes work
    /// without re-typesetting. Per-item color overrides (e.g. from
    /// `\color{red}`) are honored.
    pub fn paint(
        &self,
        origin: Point<Pixels>,
        em_px: Pixels,
        color: Hsla,
        window: &mut Window,
        cx: &mut App,
    ) {
        let em = f32::from(em_px);
        for item in &self.list.items {
            match item {
                DisplayItem::GlyphPath {
                    x,
                    y,
                    scale,
                    font,
                    char_code,
                    color: item_color,
                    ..
                } => {
                    paint_glyph(
                        window,
                        cx,
                        origin,
                        em,
                        *x as f32,
                        *y as f32,
                        *scale as f32,
                        font,
                        *char_code,
                        resolve_color(item_color, color),
                    );
                }
                DisplayItem::Line {
                    x,
                    y,
                    width,
                    thickness,
                    color: item_color,
                    dashed: _,
                } => {
                    let thick_px = (*thickness as f32 * em).max(1.0);
                    let bounds = Bounds {
                        origin: point(
                            origin.x + px(*x as f32 * em),
                            origin.y + px(*y as f32 * em - thick_px * 0.5),
                        ),
                        size: size(px(*width as f32 * em), px(thick_px)),
                    };
                    window.paint_quad(fill(bounds, resolve_color(item_color, color)));
                }
                DisplayItem::Rect {
                    x,
                    y,
                    width,
                    height,
                    color: item_color,
                } => {
                    let bounds = Bounds {
                        origin: point(origin.x + px(*x as f32 * em), origin.y + px(*y as f32 * em)),
                        size: size(px(*width as f32 * em), px(*height as f32 * em)),
                    };
                    window.paint_quad(fill(bounds, resolve_color(item_color, color)));
                }
                DisplayItem::Path {
                    x,
                    y,
                    commands,
                    fill: should_fill,
                    color: item_color,
                } => {
                    paint_path_commands(
                        window,
                        origin,
                        em,
                        *x as f32,
                        *y as f32,
                        commands,
                        *should_fill,
                        resolve_color(item_color, color),
                    );
                }
            }
        }
    }
}

/// Default black (`Color { r:0, g:0, b:0, a:1 }`) → use the theme
/// color so dark mode picks up white text without re-typesetting.
/// Anything else (e.g. `\color{red}`) is honored.
fn resolve_color(item: &ratex_types::Color, fallback: Hsla) -> Hsla {
    if item.r == 0.0 && item.g == 0.0 && item.b == 0.0 && item.a == 1.0 {
        return fallback;
    }
    let r = (item.r.clamp(0.0, 1.0) * 255.0) as u8;
    let g = (item.g.clamp(0.0, 1.0) * 255.0) as u8;
    let b = (item.b.clamp(0.0, 1.0) * 255.0) as u8;
    let a = (item.a.clamp(0.0, 1.0) * 255.0) as u8;
    let rgba = ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32);
    gpui::rgba(rgba).into()
}

/// Shape a single math glyph through gpui's text system and paint it
/// at the laid-out position. RaTeX's `font` field is one of the KaTeX
/// face names ("Main-Regular", "Math-Italic", "Size4-Regular", …);
/// we split it into family + weight + italic and request the closest
/// loaded face. `char_code` is the *display* codepoint — for the
/// Mathematical Alphanumeric Symbols range, that's not where the
/// glyph actually lives in the TTF, so we route through
/// [`ratex_font::math_alpha::katex_ttf_glyph_char`] which maps to the
/// ASCII slot the .ttf cmap actually uses.
#[allow(clippy::too_many_arguments)]
fn paint_glyph(
    window: &mut Window,
    cx: &mut App,
    origin: Point<Pixels>,
    em: f32,
    x_em: f32,
    y_em: f32,
    scale: f32,
    font_name: &str,
    char_code: u32,
    color: Hsla,
) {
    let font = make_gpui_font(font_name);
    let ratex_font_id =
        ratex_font::FontId::parse(font_name).unwrap_or(ratex_font::FontId::MainRegular);
    let ch = ratex_font::math_alpha::katex_ttf_glyph_char(ratex_font_id, char_code);
    let glyph_em = (em * scale).max(1.0);
    let glyph_em_px = px(glyph_em);
    // Resolve the gpui Font to a FontId so we can read the actual
    // ascent below — the heuristic-baseline approach (ascent ≈
    // 0.78em) was close-enough for body text but visibly off for
    // math fonts whose ascent runs nearer 0.85em (e.g. KaTeX_Size4
    // for stretched delimiters).
    let gpui_font_id = window.text_system().resolve_font(&font);
    let mut s = String::new();
    s.push(ch);
    let runs = [TextRun {
        len: s.len(),
        font,
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    }];
    let shaped = window
        .text_system()
        .shape_text(SharedString::from(s), glyph_em_px, &runs, None, None)
        .ok()
        .and_then(|mut v| v.drain(..).next());
    let Some(line) = shaped else {
        return;
    };
    // RaTeX's `y` is the glyph baseline; gpui's `WrappedLine::paint`
    // origin is the *top* of the shaped line, so offset upward by
    // the font's actual ascent.
    let ascent = window.text_system().ascent(gpui_font_id, glyph_em_px);
    let glyph_top = origin.y + px(y_em * em) - ascent;
    let glyph_left = origin.x + px(x_em * em);
    let _ = line.paint(
        point(glyph_left, glyph_top),
        px(glyph_em),
        gpui::TextAlign::Left,
        None,
        window,
        cx,
    );
}

/// Translate a RaTeX face name ("Main-Regular", "Math-Italic", etc.)
/// into a gpui `Font`. The bundled KaTeX TTFs register under
/// `family = "KaTeX_<First>"` with subfamily distinguishing
/// weight/italic, so e.g. `Main-Bold` maps to family `KaTeX_Main`,
/// weight = BOLD, style = Normal.
fn make_gpui_font(font_name: &str) -> gpui::Font {
    // First component before `-` is the family root; the rest is style.
    let (root, style) = match font_name.split_once('-') {
        Some((r, s)) => (r, s),
        None => (font_name, "Regular"),
    };
    let family = SharedString::from(format!("KaTeX_{root}"));
    let (weight, italic) = match style {
        "Bold" => (FontWeight::BOLD, false),
        "Italic" => (FontWeight::NORMAL, true),
        "BoldItalic" => (FontWeight::BOLD, true),
        _ => (FontWeight::NORMAL, false),
    };
    gpui::Font {
        family,
        features: gpui::FontFeatures::default(),
        fallbacks: None,
        weight,
        style: if italic {
            FontStyle::Italic
        } else {
            FontStyle::Normal
        },
    }
}

/// Convert a RaTeX SVG-style path (radical signs, large delimiters)
/// into a gpui `Path<Pixels>` and paint it filled. Cubic Bézier
/// commands are flattened to two quadratics each — gpui's path API
/// only supports moves, lines, and quadratic curves, and that
/// approximation is visually indistinguishable for the smooth curves
/// in math chrome at body font sizes.
#[allow(clippy::too_many_arguments)]
fn paint_path_commands(
    window: &mut Window,
    origin: Point<Pixels>,
    em: f32,
    x_em: f32,
    y_em: f32,
    commands: &[PathCommand],
    _fill: bool,
    color: Hsla,
) {
    if commands.is_empty() {
        return;
    }
    let to_px = |x: f64, y: f64| -> Point<Pixels> {
        point(
            origin.x + px((x_em + x as f32) * em),
            origin.y + px((y_em + y as f32) * em),
        )
    };
    let mut path: Option<Path<Pixels>> = None;
    let mut last = point(px(0.0), px(0.0));
    for cmd in commands {
        match *cmd {
            PathCommand::MoveTo { x, y } => {
                let p = to_px(x, y);
                if path.is_none() {
                    path = Some(Path::new(p));
                } else if let Some(ref mut pp) = path {
                    pp.move_to(p);
                }
                last = p;
            }
            PathCommand::LineTo { x, y } => {
                let p = to_px(x, y);
                if let Some(ref mut pp) = path {
                    pp.line_to(p);
                }
                last = p;
            }
            PathCommand::QuadTo { x1, y1, x, y } => {
                let p = to_px(x, y);
                let ctrl = to_px(x1, y1);
                if let Some(ref mut pp) = path {
                    pp.curve_to(p, ctrl);
                }
                last = p;
            }
            PathCommand::CubicTo {
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            } => {
                // Flatten cubic [P0, C1, C2, P3] → two quadratics.
                let p0 = last;
                let c1 = to_px(x1, y1);
                let c2 = to_px(x2, y2);
                let p3 = to_px(x, y);
                let mid = point(
                    px((f32::from(p0.x)
                        + 3.0 * f32::from(c1.x)
                        + 3.0 * f32::from(c2.x)
                        + f32::from(p3.x))
                        / 8.0),
                    px((f32::from(p0.y)
                        + 3.0 * f32::from(c1.y)
                        + 3.0 * f32::from(c2.y)
                        + f32::from(p3.y))
                        / 8.0),
                );
                let q1 = point(
                    px((f32::from(p0.x) + 3.0 * f32::from(c1.x)) / 4.0),
                    px((f32::from(p0.y) + 3.0 * f32::from(c1.y)) / 4.0),
                );
                let q2 = point(
                    px((3.0 * f32::from(c2.x) + f32::from(p3.x)) / 4.0),
                    px((3.0 * f32::from(c2.y) + f32::from(p3.y)) / 4.0),
                );
                if let Some(ref mut pp) = path {
                    pp.curve_to(mid, q1);
                    pp.curve_to(p3, q2);
                }
                last = p3;
            }
            PathCommand::Close => {
                // gpui's Path is implicitly closed by the renderer
                // when filled. Nothing to emit.
            }
        }
    }
    if let Some(p) = path {
        window.paint_path(p, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typeset_simple_inline_expression() {
        let layout = typeset("x^2", MathMode::Inline).expect("typeset succeeds");
        // Every plausible math expression has at least one display item.
        assert!(!layout.list.items.is_empty());
        // Width and height non-zero.
        assert!(layout.list.width > 0.0);
        assert!(layout.list.height + layout.list.depth > 0.0);
    }

    #[test]
    fn typeset_display_expression_has_larger_dimensions_than_inline() {
        // A fraction at display style is taller / wider than at text style.
        let inline = typeset(r"\frac{a}{b}", MathMode::Inline).unwrap();
        let display = typeset(r"\frac{a}{b}", MathMode::Display).unwrap();
        let inline_h = inline.list.height + inline.list.depth;
        let display_h = display.list.height + display.list.depth;
        assert!(
            display_h >= inline_h,
            "display height ({display_h}) should be >= inline height ({inline_h})"
        );
    }

    #[test]
    fn typeset_invalid_latex_returns_error() {
        // A clearly malformed expression — `\frac` requires two args.
        let result = typeset(r"\frac{a}", MathMode::Inline);
        assert!(result.is_err());
    }

    #[test]
    fn make_gpui_font_maps_known_faces() {
        let f = make_gpui_font("Main-Bold");
        assert_eq!(f.family.as_ref(), "KaTeX_Main");
        assert_eq!(f.weight, FontWeight::BOLD);
        assert_eq!(f.style, FontStyle::Normal);

        let f = make_gpui_font("Math-Italic");
        assert_eq!(f.family.as_ref(), "KaTeX_Math");
        assert_eq!(f.weight, FontWeight::NORMAL);
        assert_eq!(f.style, FontStyle::Italic);

        let f = make_gpui_font("Size3-Regular");
        assert_eq!(f.family.as_ref(), "KaTeX_Size3");
    }
}
