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
    /// Display-text substitutions: each entry replaces a byte range
    /// of source with a literal display string in the shaped line.
    /// Used today to render unordered list markers (`- `, `* `, `+ `)
    /// as a bullet glyph (`• `) when the cursor is outside the item.
    /// All display bytes of a substitution map back to
    /// `source_range.start` for cursor-position purposes, so a
    /// click on the bullet lands at the start of the original
    /// marker bytes.
    pub substitutions: Vec<Substitution>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Substitution {
    pub source_range: Range<usize>,
    pub display: String,
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
    ///
    /// `marker_byte_len` is this item's specific marker length in
    /// source bytes (e.g. `2` for `- `, `4` for `24. `). Used to
    /// detect and hide the matching continuation indent on subsequent
    /// lines of the item.
    ///
    /// `list_max_marker_text` is the widest marker text anywhere in
    /// the parent list (e.g. `"31. "` for an ordered list ending at
    /// item 31, `"- "` for any unordered list). The element layer
    /// shapes this string in the body font and uses its width as
    /// every sibling's content-edge indent, so all items in the list
    /// align at the same column regardless of their own marker width.
    ListItem {
        cursor_inside: bool,
        kind: ListItemKind,
        marker_byte_len: usize,
        list_max_marker_text: String,
    },
}

/// What kind of list item this is — the marker shape that produced it.
/// Used to choose the next item's marker text on Enter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListItemKind {
    /// Bullet item; the byte is the bullet character (`-`, `*`, or `+`).
    /// `task` is `Some(checked)` for GFM task list items
    /// (`- [ ] todo` / `- [x] done`); the renderer paints a checkbox
    /// glyph in place of the bullet when the cursor is outside.
    Unordered(u8, Option<bool>),
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
            substitutions: Vec::new(),
        }
    }

    /// True if `range` is hidden — either as an exact entry in
    /// `hidden_ranges` or as a sub-range of one. The post-render
    /// `merge_hidden_ranges` pass collapses overlapping / touching
    /// entries (so a multi-pass hide writing `[0..2]` and `[2..4]`
    /// canonicalizes to `[0..4]`); callers that asserted on the
    /// original sub-ranges still want a positive answer.
    pub fn has_hidden_range(&self, range: Range<usize>) -> bool {
        if range.is_empty() {
            return self.hidden_ranges.contains(&range);
        }
        self.hidden_ranges
            .iter()
            .any(|r| r.start <= range.start && range.end <= r.end)
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
    /// Thematic break — `---` / `***` / `___` on its own line.
    /// Rendered as a thin horizontal rule painted across the
    /// content width. The source bytes are hidden when the cursor
    /// is outside the construct and dimmed when inside, mirroring
    /// the delimiter rule used elsewhere.
    ThematicBreak,
    /// Display math block (`$$ ... $$`). Promoted from a paragraph
    /// whose sole content-bearing child is a `DisplayMath` event.
    /// The block has two rendering modes that swap based on cursor
    /// position:
    ///
    /// * **Display mode** (`edit_mode == false`) — cursor is *not*
    ///   strictly inside the math content range. The element layer
    ///   typesets `source[content_range]` via [`crate::math`] and
    ///   paints the rendered math; no shaped text rows.
    /// * **Edit mode** (`edit_mode == true`) — cursor is strictly
    ///   inside the construct. The block falls back to text
    ///   shaping: `$$` delimiters dim, inner LaTeX shapes in the
    ///   mono font so the user can edit it directly. (Future
    ///   iteration will reuse code-block layout for full-width
    ///   background; v1 keeps it simple.)
    ///
    /// `content_range` is the inner LaTeX bytes (between the
    /// `$$`-delimiter pairs).
    DisplayMath {
        content_range: Range<usize>,
        edit_mode: bool,
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
    /// Inline code — render in the mono font with a faint
    /// background. Set on the content of an `` `code` `` span.
    pub code: bool,
    /// Inline link text — render with the link color and an
    /// underline. Set on the content of a `[text](url)` span.
    pub link: bool,
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

    pub fn code() -> Self {
        Self {
            code: true,
            ..Self::default()
        }
    }

    pub fn link() -> Self {
        Self {
            link: true,
            ..Self::default()
        }
    }

    pub fn merge(mut self, other: InlineStyle) -> Self {
        self.bold |= other.bold;
        self.italic |= other.italic;
        self.strikethrough |= other.strikethrough;
        self.dimmed |= other.dimmed;
        self.code |= other.code;
        self.link |= other.link;
        self
    }

    pub fn is_default(&self) -> bool {
        !self.bold
            && !self.italic
            && !self.strikethrough
            && !self.dimmed
            && !self.code
            && !self.link
    }
}
