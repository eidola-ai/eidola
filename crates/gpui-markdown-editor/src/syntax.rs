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
    SoftBreak,
    HardBreak,
    Text,
}
