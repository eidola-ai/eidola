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
    /// Source ranges of *whole lines* that should be treated as
    /// delimiter (fence) lines for layout purposes — they reserve
    /// vertical space, paint outside the content scroll mask, and
    /// don't translate with horizontal scroll. Code blocks list the
    /// opener and closer fence lines here; other constructs leave
    /// it empty.
    ///
    /// This is separate from `hidden_ranges` because a fence row
    /// can have *partial* visibility (e.g. ` ```rust ` shows the
    /// `rust` info string when the cursor is outside the construct
    /// but hides the ` ``` `) — the line is still a fence row even
    /// though its full extent isn't covered by a hidden range.
    pub delimiter_lines: Vec<Range<usize>>,
    /// Container chain this leaf block sits inside, outermost first.
    /// Empty for a top-level block; `[BlockQuote, BlockQuote]` for a
    /// paragraph inside two nested blockquotes; `[BlockQuote,
    /// ListItem, …]` once lists land. The element layer reads this
    /// to compute cumulative left indent and to paint per-level
    /// decorations (blockquote borders, list markers).
    ///
    /// One leaf block per visual block; the container chain is per
    /// leaf. Sibling leaves of the same container repeat the chain
    /// — that's intentional, so `inject_empty_paragraphs` (which
    /// works on a flat list) doesn't have to know about nesting.
    pub containers: Vec<Container>,
    /// Per-line container-marker glyphs to paint as overlays on top
    /// of their corresponding container decoration (e.g. the `>` of a
    /// blockquote, painted over the level's left border bar) when the
    /// cursor is "inside" that container. Always-hidden in the shaped
    /// line itself (the marker bytes also appear in `hidden_ranges`)
    /// so the content's horizontal position is identical regardless of
    /// cursor focus — the overlay simply appears or disappears.
    pub marker_overlays: Vec<MarkerOverlay>,
}

/// A container-level glyph drawn on top of the container's
/// decoration. Today only blockquote `>` markers populate this, one
/// per blockquote-prefixed source line per cursor-inside level.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkerOverlay {
    /// Source range of the marker bytes in the buffer (e.g. `> ` or
    /// just `>`). The element layer locates the shaped line that
    /// contains `source_range.start` to choose the y position.
    pub source_range: Range<usize>,
    /// Index into `containers` of the container this marker belongs
    /// to (outermost = 0). Drives the x position so each level's
    /// overlay sits over its own border bar.
    pub level: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Container {
    /// One blockquote level wrapping this leaf. `cursor_inside` reflects
    /// the cursor's position vs. the *blockquote's* source range (not
    /// the leaf's), so all leaves of the same blockquote agree.
    BlockQuote { cursor_inside: bool },
    /// One list-item wrapping this leaf. Lists themselves contribute no
    /// chrome — the item is the visible unit. `cursor_inside` reflects
    /// the cursor's position vs. the item's source range. `kind` lets
    /// the editor choose what to insert when the user presses Enter at
    /// the end of an item (next bullet vs next number).
    ListItem {
        cursor_inside: bool,
        kind: ListItemKind,
    },
}

/// What kind of list item this is — the marker shape that produced it.
/// Used to choose the next item's marker text on Enter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListItemKind {
    /// Bullet item; the byte is the bullet character (`-`, `*`, or `+`).
    Unordered(u8),
    /// Numbered item. `number` is *this* item's parsed number; the next
    /// item produced by Enter is `number + 1`. Renumbering of later
    /// items in the list is not yet implemented.
    Ordered { number: u64 },
}

impl RenderBlock {
    pub fn new(source_range: Range<usize>, kind: BlockKind) -> Self {
        Self {
            source_range,
            kind,
            inlines: Vec::new(),
            hidden_ranges: Vec::new(),
            delimiter_lines: Vec::new(),
            containers: Vec::new(),
            marker_overlays: Vec::new(),
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

    pub fn has_marker_overlay(&self, range: Range<usize>, level: usize) -> bool {
        self.marker_overlays
            .iter()
            .any(|o| o.source_range == range && o.level == level)
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
