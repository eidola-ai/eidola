//! Visual style for the editor — fonts, sizes, paragraph spacing, and the
//! handful of colors needed for rendering. All colors are pulled from
//! `gpui_component::Theme` so Day / Night switching is automatic.
//!
//! Caller-tunable knobs mirror `gpui_component::TextViewStyle` so the chat
//! transcript renderer and the editor stay in lockstep when configured
//! identically.

use std::sync::Arc;

use gpui::{App, FontWeight, Hsla, Pixels, Rems, SharedString, px, rems};
use gpui_component::Theme;

/// Function: heading level (1..=6) + base font size → final heading size.
pub type HeadingFontSize = Arc<dyn Fn(u8, Pixels) -> Pixels + Send + Sync + 'static>;

/// Function: heading level (1..=6) → font weight. Overrides the default
/// bold-h1/h2, semibold-h3+ ramp when set.
pub type HeadingWeight = Arc<dyn Fn(u8) -> FontWeight + Send + Sync + 'static>;

#[derive(Clone)]
pub struct MarkdownStyle {
    pub font_family: SharedString,
    pub mono_font_family: SharedString,
    pub font_size: Pixels,
    pub line_height: Rems,
    pub paragraph_gap: Rems,
    /// Extra vertical breathing room added at the boundary between
    /// two blocks whose container chains don't match (e.g. paragraph
    /// → blockquote, or moving in / out of a nested level). Split
    /// half-and-half between the two adjacent blocks. Painted
    /// *outside* the blockquote border bar so the bar doesn't extend
    /// into the breathing room — the extra reads as visual
    /// separation from surrounding prose, not as part of the
    /// quoted region. Set to `rems(0.0)` for the original flush
    /// behavior.
    pub container_boundary_gap: Rems,

    /// Base size for headings. The `heading_font_size` callback (if any)
    /// scales this per level. Default is `font_size`.
    pub heading_base_font_size: Pixels,
    pub heading_font_size: Option<HeadingFontSize>,

    /// Per-level heading weight. When `None`, the default ramp applies
    /// (h1/h2 bold, h3+ semibold — weight carries hierarchy). Set it to
    /// impose a uniform or custom weight (e.g. the prose surface pins every
    /// heading to Medium and lets the size ramp carry hierarchy instead).
    pub heading_weight: Option<HeadingWeight>,

    /// Mono font size used for code blocks. Defaults to the theme's
    /// `mono_font_size`.
    pub mono_font_size: Pixels,
    /// Background of the rounded outer code-block fill. The fence
    /// rows (opening / closing fences) sit on this bg; the content
    /// area gets `code_block_content_background` painted over it as
    /// an inset strip. Defaults to `theme.muted`.
    pub code_block_background: Hsla,
    /// Background of the inner content strip — slightly darker than
    /// `code_block_background` so the code area reads as inset
    /// inside the fence frame. Defaults to a 4% darker shade of
    /// `code_block_background`.
    pub code_block_content_background: Hsla,
    /// Inner padding (top, right, bottom, left equal) inside the code
    /// block fill, before content shaping.
    pub code_block_padding: Pixels,
    /// Vertical breathing room between the fence rows and the content
    /// area inside the code-block content strip. Inserted both above
    /// the first content line and below the last so the code text
    /// doesn't sit flush against the fence rows.
    pub code_block_content_padding_y: Pixels,
    /// Corner radius of the code-block fill. Defaults to the theme's
    /// `radius`.
    pub code_block_radius: Pixels,

    /// Total horizontal indent contributed by one blockquote level —
    /// applied to the leaf content's left edge. Includes both the
    /// border-bar width and the gap between the bar and content.
    /// Nested blockquotes apply this indent cumulatively, one per
    /// level.
    pub blockquote_indent: Pixels,
    /// Vertical gap between blocks that live inside a list item,
    /// expressed as a fraction of `paragraph_gap`. List items should
    /// read as lines within one block — visibly tighter than the
    /// paragraph rhythm — with the full `paragraph_gap` (plus
    /// `container_boundary_gap`) only at the boundary between the
    /// list and its non-list neighbors. 1.0 reproduces the old
    /// "every item is a paragraph" spacing.
    pub list_item_gap_factor: f32,
    /// Horizontal indent contributed by one list-item container —
    /// applied to the leaf content's left edge so each item's content
    /// (marker plus body) sits inset from the surrounding prose. The
    /// marker glyph itself sits at the start of this indent and is
    /// part of the shaped line, so a longer marker (`12.`) and a
    /// shorter one (`-`) currently occupy different fractions of the
    /// indent — visual alignment of the marker column across items is
    /// a future polish item.
    pub list_indent: Pixels,
    /// Width of the per-level left border bar painted at the start of
    /// the indent block. The bar sits at the level's left edge; the
    /// content sits `blockquote_indent` further right.
    pub blockquote_border_width: Pixels,
    /// Horizontal inset for the *outermost* (level 0) border bar so it
    /// doesn't sit flush against the editor's leading edge. Inner
    /// nested bars inherit the same inset by virtue of being painted
    /// at `blockquote_border_inset + level * blockquote_indent`.
    pub blockquote_border_inset: Pixels,
    /// Color of the per-level left border bar. Defaults to the
    /// theme's `border` so the bar reads as chrome rather than
    /// content.
    pub blockquote_border_color: Hsla,

    pub text_color: Hsla,
    pub delimiter_color: Hsla,
    pub background: Hsla,
    pub caret_color: Hsla,
    pub selection_color: Hsla,
    /// Wash painted behind host-supplied highlight ranges on
    /// [`crate::highlight::HighlightLayer::Base`] (see [`crate::highlight`])
    /// — a quiet warm underlay, deliberately fainter than the selection so a
    /// selection over highlighted text still reads. The default is a
    /// low-alpha amber derived from the theme's mode (the same warm family in
    /// day and night); hosts can override per palette.
    pub highlight_color: Hsla,
    /// Wash for [`crate::highlight::HighlightLayer::Overlay`] — the same warm
    /// family, one step stronger, so a range on the layer above the base
    /// reads as a distinct decoration rather than a deeper merge.
    pub highlight_overlay_color: Hsla,
    /// Wash for [`crate::highlight::HighlightLayer::Accent`] — the strongest
    /// of the three, for the one range a host singles out among many.
    pub highlight_accent_color: Hsla,

    /// Font family for *inline* code spans. Defaults to
    /// `mono_font_family`, but is exposed separately because inline
    /// code sits on the same shaped line as body text: gpui's
    /// `TextRun` carries no per-run font size, so an inline span
    /// cannot be sized independently of its line (~0.9× of body is
    /// what good typography wants when the body is a serif). The
    /// achievable lever is the *family* — hosts pairing a low-x-height
    /// body face (e.g. a bookish serif) with a tall mono (Menlo) can
    /// point this at an x-height-compatible mono so inline code reads
    /// at the body's visual size, while fenced blocks (whole lines of
    /// mono, no serif neighbors) keep `mono_font_family`.
    pub inline_code_font_family: SharedString,
    /// Background color of an inline code span. Defaults to the
    /// theme's `accent` — the same chip color `gpui-component`'s
    /// `TextView` paints behind inline code in the chat transcript,
    /// keeping the two surfaces in lockstep.
    pub inline_code_background: Hsla,
    /// Text color for inline link text. Defaults to the theme's
    /// `primary` (the accent color used for actionable text in
    /// `gpui-component`).
    pub link_color: Hsla,
    /// Color used to paint the rule line of a thematic break.
    /// Defaults to the theme's `border` so rules read as chrome
    /// rather than content.
    pub thematic_break_color: Hsla,
    /// Thickness (in px) of the thematic-break rule line.
    pub thematic_break_thickness: Pixels,
    /// Font weight for table header cells. Defaults to `MEDIUM` —
    /// the same flat, quiet emphasis the app's prose headings use
    /// (weight carries the header, never size), in the book idiom of
    /// hairline-ruled tables.
    pub table_header_weight: FontWeight,
    /// Color of the table's hairline rules (under the header row and
    /// between body rows). Defaults to the theme's `border`, like
    /// the thematic break — table rules are chrome, not content.
    pub table_rule_color: Hsla,
}

impl MarkdownStyle {
    /// Build a style anchored to the active `gpui_component::Theme`.
    pub fn from_theme(cx: &App) -> Self {
        let theme = Theme::global(cx);
        Self {
            font_family: theme.font_family.clone(),
            mono_font_family: theme.mono_font_family.clone(),
            font_size: theme.font_size,
            line_height: rems(1.5),
            paragraph_gap: rems(1.0),
            container_boundary_gap: rems(0.5),

            heading_base_font_size: theme.font_size,
            heading_font_size: None,
            heading_weight: None,

            mono_font_size: theme.mono_font_size,
            code_block_background: theme.muted,
            code_block_content_background: shift_lightness(theme.muted, -0.04),
            code_block_padding: px(12.0),
            code_block_content_padding_y: px(12.0),
            code_block_radius: theme.radius,

            blockquote_indent: px(20.0),
            blockquote_border_width: px(3.0),
            blockquote_border_inset: px(6.0),
            blockquote_border_color: theme.border,

            list_item_gap_factor: 0.35,
            list_indent: px(8.0),

            text_color: theme.foreground,
            delimiter_color: theme.muted_foreground,
            background: theme.background,
            caret_color: theme.caret,
            selection_color: theme.selection,
            highlight_color: if theme.mode.is_dark() {
                gpui::hsla(0.115, 0.55, 0.55, 0.18)
            } else {
                gpui::hsla(0.115, 0.85, 0.55, 0.16)
            },
            highlight_overlay_color: if theme.mode.is_dark() {
                gpui::hsla(0.115, 0.55, 0.55, 0.32)
            } else {
                gpui::hsla(0.115, 0.85, 0.55, 0.30)
            },
            highlight_accent_color: if theme.mode.is_dark() {
                gpui::hsla(0.115, 0.70, 0.60, 0.55)
            } else {
                gpui::hsla(0.115, 0.90, 0.55, 0.52)
            },

            inline_code_font_family: theme.mono_font_family.clone(),
            inline_code_background: theme.accent,
            link_color: theme.link,
            thematic_break_color: theme.border,
            thematic_break_thickness: px(1.0),
            table_header_weight: FontWeight::MEDIUM,
            table_rule_color: theme.border,
        }
    }

    pub fn font_size(mut self, size: Pixels) -> Self {
        self.font_size = size;
        self.heading_base_font_size = size;
        self
    }

    pub fn paragraph_gap(mut self, gap: Rems) -> Self {
        self.paragraph_gap = gap;
        self
    }

    pub fn container_boundary_gap(mut self, gap: Rems) -> Self {
        self.container_boundary_gap = gap;
        self
    }

    pub fn line_height(mut self, height: Rems) -> Self {
        self.line_height = height;
        self
    }

    pub fn heading_base_font_size(mut self, size: Pixels) -> Self {
        self.heading_base_font_size = size;
        self
    }

    pub fn heading_font_size<F>(mut self, f: F) -> Self
    where
        F: Fn(u8, Pixels) -> Pixels + Send + Sync + 'static,
    {
        self.heading_font_size = Some(Arc::new(f));
        self
    }

    pub fn heading_weight<F>(mut self, f: F) -> Self
    where
        F: Fn(u8) -> FontWeight + Send + Sync + 'static,
    {
        self.heading_weight = Some(Arc::new(f));
        self
    }

    pub fn mono_font_size(mut self, size: Pixels) -> Self {
        self.mono_font_size = size;
        self
    }

    pub fn code_block_padding(mut self, pad: Pixels) -> Self {
        self.code_block_padding = pad;
        self
    }

    pub fn code_block_content_padding_y(mut self, pad: Pixels) -> Self {
        self.code_block_content_padding_y = pad;
        self
    }

    pub fn code_block_background(mut self, bg: Hsla) -> Self {
        self.code_block_background = bg;
        self
    }

    pub fn code_block_content_background(mut self, bg: Hsla) -> Self {
        self.code_block_content_background = bg;
        self
    }

    pub fn code_block_radius(mut self, radius: Pixels) -> Self {
        self.code_block_radius = radius;
        self
    }

    pub fn blockquote_indent(mut self, indent: Pixels) -> Self {
        self.blockquote_indent = indent;
        self
    }

    pub fn blockquote_border_width(mut self, width: Pixels) -> Self {
        self.blockquote_border_width = width;
        self
    }

    pub fn blockquote_border_inset(mut self, inset: Pixels) -> Self {
        self.blockquote_border_inset = inset;
        self
    }

    pub fn blockquote_border_color(mut self, color: Hsla) -> Self {
        self.blockquote_border_color = color;
        self
    }

    pub fn list_indent(mut self, indent: Pixels) -> Self {
        self.list_indent = indent;
        self
    }

    pub fn list_item_gap_factor(mut self, factor: f32) -> Self {
        self.list_item_gap_factor = factor;
        self
    }

    pub fn inline_code_font_family(mut self, family: impl Into<SharedString>) -> Self {
        self.inline_code_font_family = family.into();
        self
    }

    pub fn inline_code_background(mut self, bg: Hsla) -> Self {
        self.inline_code_background = bg;
        self
    }

    /// Override the base layer's highlight wash color (see
    /// `highlight_color`).
    pub fn highlight_color(mut self, color: Hsla) -> Self {
        self.highlight_color = color;
        self
    }

    /// Override the overlay layer's wash color (see
    /// `highlight_overlay_color`).
    pub fn highlight_overlay_color(mut self, color: Hsla) -> Self {
        self.highlight_overlay_color = color;
        self
    }

    /// Override the accent layer's wash color (see `highlight_accent_color`).
    pub fn highlight_accent_color(mut self, color: Hsla) -> Self {
        self.highlight_accent_color = color;
        self
    }

    /// The wash color for one highlight layer. Total by construction — every
    /// [`crate::highlight::HighlightLayer`] has a color, so a host that paints
    /// on a layer can never get an unstyled wash.
    pub fn highlight_layer_color(&self, layer: crate::highlight::HighlightLayer) -> Hsla {
        match layer {
            crate::highlight::HighlightLayer::Base => self.highlight_color,
            crate::highlight::HighlightLayer::Overlay => self.highlight_overlay_color,
            crate::highlight::HighlightLayer::Accent => self.highlight_accent_color,
        }
    }

    /// Final font size for `level` (1..=6). Uses the callback if set,
    /// otherwise a sensible default.
    pub fn size_for_heading(&self, level: u8) -> Pixels {
        let base = self.heading_base_font_size;
        if let Some(f) = &self.heading_font_size {
            return f(level, base);
        }
        let mult: f32 = match level {
            1 => 1.5,
            2 => 1.25,
            3 => 1.125,
            _ => 1.0,
        };
        px(f32::from(base) * mult)
    }

    /// Final font weight for `level` (1..=6). Uses the `heading_weight`
    /// callback if set, otherwise the default ramp (h1/h2 bold, h3+ semibold).
    pub fn weight_for_heading(&self, level: u8) -> FontWeight {
        if let Some(f) = &self.heading_weight {
            return f(level);
        }
        if self.heading_is_bold(level) {
            FontWeight::BOLD
        } else {
            FontWeight::SEMIBOLD
        }
    }

    /// h1 / h2 are bold; h3+ are semibold. Only consulted for the default
    /// weight ramp — a `heading_weight` override supersedes it.
    pub fn heading_is_bold(&self, level: u8) -> bool {
        level <= 2
    }
}

/// Shift the lightness of an HSLA color by `delta` (in the 0..=1
/// space), clamping to the valid range. Negative values darken,
/// positive values lighten. Used to derive the code-block content
/// strip color from the outer fill so a Day/Night theme switch keeps
/// them in proportion automatically.
pub(crate) fn shift_lightness(mut color: Hsla, delta: f32) -> Hsla {
    color.l = (color.l + delta).clamp(0.0, 1.0);
    color
}
