//! Visual style for the editor — fonts, sizes, paragraph spacing, and the
//! handful of colors needed for rendering. All colors are pulled from
//! `gpui_component::Theme` so Day / Night switching is automatic.
//!
//! Caller-tunable knobs mirror `gpui_component::TextViewStyle` so the chat
//! transcript renderer and the editor stay in lockstep when configured
//! identically.

use std::sync::Arc;

use gpui::{App, Hsla, Pixels, Rems, SharedString, px, rems};
use gpui_component::Theme;

/// Function: heading level (1..=6) + base font size → final heading size.
pub type HeadingFontSize = Arc<dyn Fn(u8, Pixels) -> Pixels + Send + Sync + 'static>;

#[derive(Clone)]
pub struct MarkdownStyle {
    pub font_family: SharedString,
    pub mono_font_family: SharedString,
    pub font_size: Pixels,
    pub line_height: Rems,
    pub paragraph_gap: Rems,

    /// Base size for headings. The `heading_font_size` callback (if any)
    /// scales this per level. Default is `font_size`.
    pub heading_base_font_size: Pixels,
    pub heading_font_size: Option<HeadingFontSize>,

    pub text_color: Hsla,
    pub delimiter_color: Hsla,
    pub background: Hsla,
    pub caret_color: Hsla,
    pub selection_color: Hsla,
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

            heading_base_font_size: theme.font_size,
            heading_font_size: None,

            text_color: theme.foreground,
            delimiter_color: theme.muted_foreground,
            background: theme.background,
            caret_color: theme.caret,
            selection_color: theme.selection,
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

    /// h1 / h2 are bold; h3+ are semibold.
    pub fn heading_is_bold(&self, level: u8) -> bool {
        level <= 2
    }
}
