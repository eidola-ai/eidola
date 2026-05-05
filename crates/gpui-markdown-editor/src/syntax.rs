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
    SoftBreak,
    HardBreak,
    Text,
}
