//! Output of `render::render` — a list of `RenderBlock`s, one per top-level
//! visual block. The element layer (`element.rs`) consumes this to shape and
//! paint each block.
//!
//! `dimmed: true` on an inline run means "delimiter visible, but rendered in
//! the theme's `delimiter_color`" — i.e. the cursor is inside the construct
//! and the user should see the raw markdown source.

use std::ops::Range;

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSpec {
    pub blocks: Vec<RenderBlock>,
}

impl RenderSpec {
    pub fn empty() -> Self {
        Self { blocks: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderBlock {
    pub source_range: Range<usize>,
    pub kind: BlockKind,
    pub inlines: Vec<InlineRun>,
    pub hidden_ranges: Vec<Range<usize>>,
}

impl RenderBlock {
    pub fn new(source_range: Range<usize>, kind: BlockKind) -> Self {
        Self {
            source_range,
            kind,
            inlines: Vec::new(),
            hidden_ranges: Vec::new(),
        }
    }

    pub fn has_hidden_range(&self, range: Range<usize>) -> bool {
        self.hidden_ranges.contains(&range)
    }

    pub fn has_dimmed_range(&self, range: Range<usize>) -> bool {
        self.inlines
            .iter()
            .any(|r| r.source_range == range && r.style.dimmed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockKind {
    Paragraph,
    Heading {
        level: u8,
    },
    /// Fenced code block. `lang` is the trimmed info string (`Some("rust")`,
    /// `Some("")` for an empty info string, or `None` for an indented
    /// block — not yet emitted, reserved). The block is a *leaf*: no
    /// inline markdown is parsed inside, the renderer ships content as
    /// raw text, and the element layer paints in the mono font with a
    /// non-wrapping shape pass.
    CodeBlock {
        lang: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct InlineRun {
    pub source_range: Range<usize>,
    pub style: InlineStyle,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct InlineStyle {
    pub bold: bool,
    pub italic: bool,
    pub strikethrough: bool,
    pub dimmed: bool,
}

impl InlineStyle {
    pub fn dimmed() -> Self {
        Self {
            dimmed: true,
            ..Self::default()
        }
    }

    pub fn bold() -> Self {
        Self {
            bold: true,
            ..Self::default()
        }
    }

    pub fn italic() -> Self {
        Self {
            italic: true,
            ..Self::default()
        }
    }

    pub fn merge(mut self, other: InlineStyle) -> Self {
        self.bold |= other.bold;
        self.italic |= other.italic;
        self.strikethrough |= other.strikethrough;
        self.dimmed |= other.dimmed;
        self
    }

    pub fn is_default(&self) -> bool {
        !self.bold && !self.italic && !self.strikethrough && !self.dimmed
    }
}
