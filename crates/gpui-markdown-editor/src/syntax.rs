//! Parsed syntax tree. Only the constructs we currently render appear here
//! — plain paragraphs / text, ATX headings, and the three inline styles.
//! Adding a construct means a new `NodeKind` variant *and* corresponding
//! `parser.rs` and `render.rs` arms.

use std::ops::Range;

pub type ByteRange = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxNode {
    pub kind: NodeKind,
    pub range: ByteRange,
    pub children: Vec<SyntaxNode>,
}

impl SyntaxNode {
    pub fn new(kind: NodeKind, range: ByteRange) -> Self {
        Self {
            kind,
            range,
            children: Vec::new(),
        }
    }

    pub fn with_children(kind: NodeKind, range: ByteRange, children: Vec<SyntaxNode>) -> Self {
        Self {
            kind,
            range,
            children,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Paragraph,
    Heading {
        level: u8,
        content_range: ByteRange,
        delimiter_ranges: Vec<ByteRange>,
    },
    Strong {
        delimiter_ranges: Vec<ByteRange>,
        content_range: ByteRange,
    },
    Emphasis {
        delimiter_ranges: Vec<ByteRange>,
        content_range: ByteRange,
    },
    Strikethrough {
        delimiter_ranges: Vec<ByteRange>,
        content_range: ByteRange,
    },
    /// Fenced code block. `delimiter_ranges` is the opening fence-char
    /// run (e.g. ` ``` `) and the closing fence-char run, both *without*
    /// the info string — those are what the renderer hides when the
    /// cursor is outside the construct. `info_string_range` is the
    /// trailing portion of the opening line *after* the fence chars
    /// (e.g. `rust` in ` ```rust `): it stays visible when the cursor
    /// is outside (so a reader can still see the language tag) but is
    /// dimmed alongside the fences when the cursor is inside.
    /// `content_range` is the inner code, excluding fence lines and
    /// the newlines that bound them.
    CodeBlock {
        lang: Option<String>,
        content_range: ByteRange,
        delimiter_ranges: Vec<ByteRange>,
        info_string_range: Option<ByteRange>,
    },
    /// Blockquote container. Each entry in `prefix_ranges` is the `> `
    /// (or `>`) marker that introduces *this* blockquote level on a
    /// single line — one per line covered by the blockquote.
    /// Outer-blockquote markers on the same line belong to the parent
    /// `BlockQuote` node's `prefix_ranges`, not this one. Mirrors
    /// `apps/macos/Packages/MarkdownEditor` `blockquotePrefixRanges`.
    /// Children carry the inner block content (paragraphs, code
    /// blocks, nested blockquotes, …).
    BlockQuote {
        prefix_ranges: Vec<ByteRange>,
    },
    /// List container. `kind` distinguishes ordered vs unordered.
    /// Children are one `ListItem` per item, in source order. The
    /// list itself contributes no rendered chrome — its presence is
    /// implied by the items it wraps.
    List {
        kind: ListKind,
    },
    /// One list item. `marker_range` covers the marker plus the
    /// trailing space (e.g. `- ` or `1. `). Children are the item's
    /// content blocks. For tight single-paragraph items pulldown
    /// emits a `Text` leaf directly; for loose items it wraps the
    /// content in `Paragraph`. The renderer treats both shapes the
    /// same — one leaf per item, with the marker hidden / dimmed
    /// according to the cursor-visibility rule.
    ListItem {
        marker_range: ByteRange,
    },
    SoftBreak,
    HardBreak,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListKind {
    /// Bullet list (`-`, `*`, `+`).
    Unordered,
    /// Numbered list. `start` is the parsed start number — preserved
    /// from source so that pasted numbered lists retain their
    /// numbering. Re-numbering on edit is not yet implemented.
    Ordered { start: u64 },
}
