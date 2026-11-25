//! Prose typography — the book metaphor every post body and every composer
//! shares, so a post renders pixel-identical to what its author typed.
//!
//! Newsreader at a book size and leading with a size-led heading ramp, distinct
//! from the system UI font the theme uses for chrome. Narrative surfaces opt
//! into it through [`prose_style`]; chrome does not.

use gpui::{FontWeight, Pixels, px, rems};
use gpui_component::ActiveTheme as _;
use gpui_markdown_editor::MarkdownStyle;

/// The prose reading size at Actual Size. Matches the composer's typography so
/// a post shapes pixel-for-pixel like what the user types.
pub const PROSE_FONT_SIZE: Pixels = px(17.);
pub const PROSE_LINE_HEIGHT: f32 = 1.65;
/// Inter-block spacing as a multiple of the font size (the editor splits it
/// half above / half below each block). The single source of truth for both
/// [`prose_style`] and any height estimate, so they can't drift.
pub const PROSE_PARAGRAPH_GAP: f32 = 1.5;

/// `MarkdownStyle` for prose bodies and the composer: Newsreader at a book
/// size/leading with a size-led heading ramp (h1 2.5× … h4 1.125×) at a
/// uniform Medium (500) weight — size carries the hierarchy — and Courier-New
/// inline code (its x-height matches Newsreader's, where Menlo reads too
/// large). Mirrors the website's prose ramp (`www/static/site.css`).
/// `from_theme` seeds the system font + theme colors, so we override the
/// family back to Newsreader for narrative content.
///
/// `font_scale` is the host's type-scale factor: the prose ramp is the one
/// place the type ramp is spelled in absolute pixels rather than `rems()`, so
/// it doesn't ride the scaled `rem_size` automatically and the base size is
/// multiplied here. The `rems()`-relative leading and paragraph gap *do* ride
/// `rem_size` (the theme UI font size, itself scaled), so they stay
/// proportional to the scaled prose size for free.
pub fn prose_style(cx: &gpui::App, font_scale: f32) -> MarkdownStyle {
    let base = PROSE_FONT_SIZE * font_scale;
    let mut style = MarkdownStyle::from_theme(cx)
        .font_size(base)
        .line_height(rems(PROSE_LINE_HEIGHT))
        .paragraph_gap(rems(PROSE_PARAGRAPH_GAP))
        .heading_base_font_size(base)
        .heading_font_size(|level, base| match level {
            1 => base * 2.5,
            2 => base * 1.75,
            3 => base * 1.25,
            _ => base * 1.125,
        })
        .heading_weight(|_| FontWeight::MEDIUM)
        .highlight_color(quoted_passage_wash(cx))
        .inline_code_font_family("Courier New");
    style.font_family = crate::FONT_FAMILY.into();
    style
}

/// The wash behind a passage someone has quoted (the editor's opaque
/// highlight plugin).
///
/// It reads as a **reader's pencil mark, not a UI selection**: the brand's own
/// warm amber at a very low alpha, so the text on top is untouched and the
/// mark is something you notice rather than something that interrupts. It is
/// deliberately fainter than `theme.selection` in both palettes, so selecting
/// across a quoted passage still reads as a selection; day carries a touch
/// less alpha than night because the near-white paper shows a wash more
/// readily than the blue-grey dark ground does.
pub fn quoted_passage_wash(cx: &gpui::App) -> gpui::Hsla {
    if cx.theme().mode.is_dark() {
        gpui::hsla(0.09, 0.50, 0.60, 0.16)
    } else {
        gpui::hsla(0.09, 0.70, 0.52, 0.13)
    }
}
