//! Walks pulldown-cmark events into a `Vec<SyntaxNode>`. Currently only the
//! variants we render produce structured nodes; everything else collapses to
//! `Paragraph` / `Text` so cursor and selection geometry still works on
//! unsupported markdown.

// `vec![start..end]` is intentionally a one-element vec of ranges (slot for
// later split delimiter ranges), not a vec containing every offset.
#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use crate::syntax::{NodeKind, SyntaxNode};

pub fn parse(markdown: &str) -> Vec<SyntaxNode> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);

    let mut walker = Walker::new(markdown);
    for (event, range) in Parser::new_ext(markdown, opts).into_offset_iter() {
        walker.handle(event, range);
    }
    walker.finish()
}

struct Frame {
    node: SyntaxNode,
}

struct Walker<'a> {
    source: &'a str,
    stack: Vec<Frame>,
    output: Vec<SyntaxNode>,
}

impl<'a> Walker<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            stack: Vec::new(),
            output: Vec::new(),
        }
    }

    fn finish(mut self) -> Vec<SyntaxNode> {
        std::mem::take(&mut self.output)
    }

    fn handle(&mut self, event: Event<'_>, range: Range<usize>) {
        match event {
            Event::Start(tag) => self.start(tag, range),
            Event::End(end) => self.end(end),
            Event::Text(_) => self.commit_leaf(SyntaxNode::new(NodeKind::Text, range)),
            Event::SoftBreak => self.commit_leaf(SyntaxNode::new(NodeKind::SoftBreak, range)),
            Event::HardBreak => self.commit_leaf(SyntaxNode::new(NodeKind::HardBreak, range)),
            // For now, anything we don't model becomes plain text so cursor
            // geometry on the source bytes still works.
            _ => self.commit_leaf(SyntaxNode::new(NodeKind::Text, range)),
        }
    }

    fn start(&mut self, tag: Tag<'_>, range: Range<usize>) {
        match tag {
            Tag::Paragraph => self.push_frame(SyntaxNode::new(NodeKind::Paragraph, range)),
            Tag::Heading { level, .. } => {
                let kind = self.heading_kind(level as u8, &range);
                self.push_frame(SyntaxNode::new(kind, range));
            }
            Tag::Emphasis => {
                let (delim, content) = self.symmetric_delimiters(&range, 1);
                self.push_frame(SyntaxNode::new(
                    NodeKind::Emphasis {
                        delimiter_ranges: delim,
                        content_range: content,
                    },
                    range,
                ));
            }
            Tag::Strong => {
                let (delim, content) = self.symmetric_delimiters(&range, 2);
                self.push_frame(SyntaxNode::new(
                    NodeKind::Strong {
                        delimiter_ranges: delim,
                        content_range: content,
                    },
                    range,
                ));
            }
            Tag::Strikethrough => {
                let (delim, content) = self.symmetric_delimiters(&range, 2);
                self.push_frame(SyntaxNode::new(
                    NodeKind::Strikethrough {
                        delimiter_ranges: delim,
                        content_range: content,
                    },
                    range,
                ));
            }
            // Unhandled containers — treat as a paragraph for now so children
            // still flow through. A future phase will add real handling.
            _ => self.push_frame(SyntaxNode::new(NodeKind::Paragraph, range)),
        }
    }

    fn end(&mut self, _end: TagEnd) {
        if let Some(frame) = self.stack.pop() {
            self.commit(frame.node);
        }
    }

    fn push_frame(&mut self, node: SyntaxNode) {
        self.stack.push(Frame { node });
    }

    fn commit_leaf(&mut self, node: SyntaxNode) {
        self.commit(node);
    }

    fn commit(&mut self, node: SyntaxNode) {
        if let Some(top) = self.stack.last_mut() {
            top.node.children.push(node);
        } else {
            self.output.push(node);
        }
    }

    fn slice(&self, r: &Range<usize>) -> &str {
        let end = r.end.min(self.source.len());
        let start = r.start.min(end);
        &self.source[start..end]
    }

    fn heading_kind(&self, level: u8, range: &Range<usize>) -> NodeKind {
        // ATX only — we don't normalize setext yet.
        let text = self.slice(range);
        let bytes = text.as_bytes();
        if bytes.first().copied() == Some(b'#') {
            let mut delim_len = 0;
            while delim_len < bytes.len() && bytes[delim_len] == b'#' {
                delim_len += 1;
            }
            while delim_len < bytes.len() && (bytes[delim_len] == b' ' || bytes[delim_len] == b'\t')
            {
                delim_len += 1;
            }
            let abs_delim = range.start..range.start + delim_len;
            let content = abs_delim.end..range.end;
            NodeKind::Heading {
                level,
                content_range: content,
                delimiter_ranges: vec![abs_delim],
            }
        } else {
            // Setext heading — minimum-viable: treat as a heading whose
            // delimiter is the underline run on the last line. We don't
            // normalize to ATX yet.
            if let Some(idx) = text.rfind('\n') {
                let content_end = range.start + idx;
                let underline_start = range.start + idx + 1;
                NodeKind::Heading {
                    level,
                    content_range: range.start..content_end,
                    delimiter_ranges: vec![underline_start..range.end],
                }
            } else {
                NodeKind::Heading {
                    level,
                    content_range: range.clone(),
                    delimiter_ranges: Vec::new(),
                }
            }
        }
    }

    fn symmetric_delimiters(
        &self,
        range: &Range<usize>,
        width: usize,
    ) -> (Vec<Range<usize>>, Range<usize>) {
        let len = range.end.saturating_sub(range.start);
        let w = width.min(len / 2);
        let opening = range.start..range.start + w;
        let closing = range.end - w..range.end;
        let content = opening.end..closing.start;
        (vec![opening, closing], content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(nodes: &[SyntaxNode]) -> &SyntaxNode {
        nodes.first().expect("at least one block")
    }

    #[test]
    fn parses_atx_heading() {
        let nodes = parse("# Hello\n");
        assert_eq!(nodes.len(), 1);
        match &first(&nodes).kind {
            NodeKind::Heading {
                level,
                content_range,
                delimiter_ranges,
            } => {
                assert_eq!(*level, 1);
                assert_eq!(delimiter_ranges.len(), 1);
                assert_eq!(delimiter_ranges[0], 0..2);
                assert_eq!(content_range.start, 2);
            }
            other => panic!("expected heading, got {other:?}"),
        }
    }

    #[test]
    fn parses_paragraph_with_strong_emphasis_and_strike() {
        let src = "a **bold** *em* ~~no~~";
        let nodes = parse(src);
        let para = first(&nodes);
        assert!(matches!(para.kind, NodeKind::Paragraph));
        let kinds: Vec<_> = para.children.iter().map(|c| &c.kind).collect();
        assert!(kinds.iter().any(|k| matches!(k, NodeKind::Strong { .. })));
        assert!(kinds.iter().any(|k| matches!(k, NodeKind::Emphasis { .. })));
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, NodeKind::Strikethrough { .. }))
        );

        for child in &para.children {
            if let NodeKind::Strong {
                delimiter_ranges,
                content_range,
            } = &child.kind
            {
                assert_eq!(&src[delimiter_ranges[0].clone()], "**");
                assert_eq!(&src[content_range.clone()], "bold");
            }
        }
    }

    #[test]
    fn empty_input_produces_no_blocks() {
        assert!(parse("").is_empty());
    }
}
