//! Walks pulldown-cmark events into a `Vec<SyntaxNode>`. Currently only the
//! variants we render produce structured nodes; everything else collapses to
//! `Paragraph` / `Text` so cursor and selection geometry still works on
//! unsupported markdown.

// `vec![start..end]` is intentionally a one-element vec of ranges (slot for
// later split delimiter ranges), not a vec containing every offset.
#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

use crate::syntax::{ListKind, NodeKind, SyntaxNode};

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
            Tag::CodeBlock(CodeBlockKind::Fenced(lang)) => {
                let kind = self.fenced_code_block_kind(&range, Some(lang.into_string()));
                self.push_frame(SyntaxNode::new(kind, range));
            }
            Tag::CodeBlock(CodeBlockKind::Indented) => {
                // Indented code blocks aren't a first-class construct yet —
                // treat them as plain paragraphs so cursor geometry still
                // works on the source bytes.
                self.push_frame(SyntaxNode::new(NodeKind::Paragraph, range));
            }
            Tag::BlockQuote(_) => {
                // Alerts (Note / Tip / Important / Warning / Caution) are
                // a GFM extension we don't render specially yet — they
                // collapse to a plain blockquote.
                let kind = self.blockquote_kind(&range);
                self.push_frame(SyntaxNode::new(kind, range));
            }
            Tag::List(start) => {
                let kind = match start {
                    Some(n) => ListKind::Ordered { start: n },
                    None => ListKind::Unordered,
                };
                self.push_frame(SyntaxNode::new(NodeKind::List { kind }, range));
            }
            Tag::Item => {
                let marker_range = self.list_item_marker_range(&range);
                self.push_frame(SyntaxNode::new(NodeKind::ListItem { marker_range }, range));
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

    /// Identify the opening / closing fence and the inner content of a
    /// fenced code block. pulldown-cmark gives us `range` covering the
    /// whole construct (including fences and any trailing `\n`), and the
    /// info string (`lang`). We scan the bytes to:
    ///
    /// * locate the opening fence — the leading run of `` ` `` or `~`,
    ///   together with any info string and the `\n` that ends the
    ///   opening line (treated as a single delimiter range that the
    ///   renderer hides outside / dims inside);
    /// * locate the closing fence — the trailing run of the same
    ///   character on its own line, together with any preceding `\n`;
    /// * everything between is `content_range`.
    ///
    /// An *unterminated* fenced block (the file ends before the closing
    /// fence) is supported: there is one delimiter range (the opener)
    /// and the content runs to `range.end`. CommonMark allows this and
    /// pulldown-cmark emits the `CodeBlock` tag even when the closing
    /// fence is absent.
    fn fenced_code_block_kind(&self, range: &Range<usize>, lang: Option<String>) -> NodeKind {
        let bytes = self.source.as_bytes();
        let start = range.start;
        let end = range.end;

        // Opening fence character: the first non-whitespace byte at the
        // start of `range`. We tolerate up to 3 leading spaces of
        // indentation per CommonMark, so scan past them first.
        let mut p = start;
        while p < end && (bytes[p] == b' ' || bytes[p] == b'\t') {
            p += 1;
        }
        let fence_char = bytes.get(p).copied().unwrap_or(b'`');

        // Opening fence run: one or more `fence_char`s.
        while p < end && bytes[p] == fence_char {
            p += 1;
        }
        // End of the fence-char run is also the start of any info
        // string. The renderer hides the fence run when the cursor
        // is outside the construct *but* keeps the info string
        // visible (so a reader sees the language tag), so the two
        // halves of the opener line need separate ranges.
        let opener_fence_end = p;
        // Info string + rest of opening line. The opener delimiter
        // ends *before* the trailing `\n` so it stays on a single
        // line, matching how the closer delimiter is shaped (also
        // pre-newline). The element layer's hidden-range lookup is
        // per-line and matches `r.end <= line_logical_end`, so an
        // opener that included the `\n` would silently fail the
        // match and render the fence text instead of hiding it.
        while p < end && bytes[p] != b'\n' {
            p += 1;
        }
        let opener_end = p;
        let after_opener_newline = if p < end && bytes[p] == b'\n' {
            p + 1
        } else {
            p
        };
        // The info string is everything between the fence run and
        // the trailing newline of the opener line. Pulldown's `lang`
        // is this same span trimmed of leading/trailing whitespace —
        // we keep the *un-trimmed* span here because the renderer
        // shapes raw bytes (a reader expects to see exactly what
        // they typed). `None` when empty, so the renderer can skip
        // dim/visibility logic for blocks without an info string.
        let info_string_range = if opener_fence_end < opener_end {
            Some(opener_fence_end..opener_end)
        } else {
            None
        };

        // Closing fence: walk back from `end` over a trailing `\n`,
        // then over a run of `fence_char`s, then over leading spaces /
        // tabs on that line. If we land on a `\n` before any
        // `fence_char` was consumed, there's no closing fence.
        let mut q = end;
        // Strip exactly one trailing `\n` if present (CommonMark allows
        // pulldown to either include or exclude it depending on whether
        // the file ends mid-block).
        if q > after_opener_newline && bytes[q - 1] == b'\n' {
            q -= 1;
        }
        // Scan back over fence chars on the closing line.
        let mut closing_fence_start = q;
        while closing_fence_start > opener_end && bytes[closing_fence_start - 1] == fence_char {
            closing_fence_start -= 1;
        }
        let has_closing_fence = closing_fence_start < q;
        // Skip indentation on the closing fence line. The `\n` that
        // ends the *previous* line stays in `content_range` — it's the
        // EOL of the last code line, not part of the closing fence.
        let mut closing_indent_start = closing_fence_start;
        if has_closing_fence {
            while closing_indent_start > after_opener_newline
                && (bytes[closing_indent_start - 1] == b' '
                    || bytes[closing_indent_start - 1] == b'\t')
            {
                closing_indent_start -= 1;
            }
        }

        let lang = lang.map(|s| s.trim().to_string());

        // The opener delimiter range covers *only* the fence-char run
        // (e.g. ` ``` `, not ` ```rust `). The info string is
        // tracked separately in `info_string_range` so the renderer
        // can hide the fence chars when the cursor is outside the
        // construct while keeping the info string visible.
        let opener_fence = start..opener_fence_end;

        if has_closing_fence {
            // The closer delimiter spans the closing fence line *up
            // to but not including* the trailing `\n`. `q` points
            // at the closing line's post-fence trailing-whitespace
            // boundary (i.e. its logical end before the trailing
            // `\n`).
            let closer = closing_indent_start..q;
            let content_range = after_opener_newline..closing_indent_start;
            NodeKind::CodeBlock {
                lang,
                content_range,
                delimiter_ranges: vec![opener_fence, closer],
                info_string_range,
            }
        } else {
            // Unterminated — one delimiter (the opener fence) and
            // content runs to the end of the parser-reported range.
            let content_range = after_opener_newline..end;
            NodeKind::CodeBlock {
                lang,
                content_range,
                delimiter_ranges: vec![opener_fence],
                info_string_range,
            }
        }
    }

    /// Build the `prefix_ranges` for a blockquote spanning `range`. Each
    /// element is the `>` (and optional trailing space) that introduces
    /// *this* blockquote level on a single line, in source order. Outer
    /// blockquote markers belong to ancestor `BlockQuote` nodes — we
    /// skip past `outer_depth` of them per line. Lazy continuation
    /// lines (paragraph continuations without a `>` for this depth)
    /// contribute no entry; the renderer treats them as regular content
    /// lines, which is what CommonMark intends.
    fn blockquote_kind(&self, range: &Range<usize>) -> NodeKind {
        let outer_depth = self
            .stack
            .iter()
            .filter(|f| matches!(f.node.kind, NodeKind::BlockQuote { .. }))
            .count();
        NodeKind::BlockQuote {
            prefix_ranges: self.blockquote_prefix_ranges(range, outer_depth),
        }
    }

    /// Locate the marker bytes (e.g. `- `, `* `, `1. `) that introduce
    /// the list item spanning `range`. Pulldown's Item range starts
    /// at the *very* first byte of the item line (any leading
    /// whitespace + the marker character(s) + the optional trailing
    /// space), so we scan forward over leading spaces, the marker
    /// run, and a single trailing space.
    ///
    /// CommonMark marker shapes:
    ///   - Bullet: one of `-`, `*`, `+`
    ///   - Ordered: one or more digits followed by `.` or `)`
    ///
    /// In both forms the marker character(s) are followed by at least
    /// one space (or tab) before the content. We consume exactly one
    /// trailing space — the rest, if any, are content indentation.
    fn list_item_marker_range(&self, range: &Range<usize>) -> Range<usize> {
        let bytes = self.source.as_bytes();
        let mut q = range.start;
        // Up to 3 leading spaces of indent precede the marker.
        let mut indent = 0;
        while q < range.end && bytes[q] == b' ' && indent < 3 {
            q += 1;
            indent += 1;
        }
        let marker_start = q;
        if q < range.end && (bytes[q] == b'-' || bytes[q] == b'*' || bytes[q] == b'+') {
            q += 1;
        } else {
            while q < range.end && bytes[q].is_ascii_digit() {
                q += 1;
            }
            if q < range.end && (bytes[q] == b'.' || bytes[q] == b')') {
                q += 1;
            }
        }
        if q < range.end && bytes[q] == b' ' {
            q += 1;
        }
        marker_start..q
    }

    fn blockquote_prefix_ranges(
        &self,
        range: &Range<usize>,
        outer_depth: usize,
    ) -> Vec<Range<usize>> {
        let bytes = self.source.as_bytes();
        let mut out = Vec::new();
        // pulldown-cmark gives a *nested* blockquote's range starting
        // mid-line (right after the outer's `>` markers), so on the
        // first line the inner range can begin at byte 2 even though
        // the source line begins at byte 0. Walking back to the prior
        // `\n` (or doc start) gives us the real line origin so the
        // outer-depth skip walks past the same `>`s the parent node
        // already claimed.
        let mut p = range.start;
        while p < range.end {
            let mut line_start = p;
            while line_start > 0 && bytes[line_start - 1] != b'\n' {
                line_start -= 1;
            }
            let mut line_end = p.max(line_start);
            while line_end < bytes.len() && bytes[line_end] != b'\n' {
                line_end += 1;
            }
            let mut q = line_start;
            let mut found = 0;
            // CommonMark allows up to 3 leading spaces of indent before
            // each `>`, plus an optional single trailing space the
            // marker consumes.
            loop {
                let mut indent = 0;
                while q < line_end && bytes[q] == b' ' && indent < 3 {
                    q += 1;
                    indent += 1;
                }
                if q < line_end && bytes[q] == b'>' {
                    let marker_start = q;
                    q += 1;
                    if q < line_end && bytes[q] == b' ' {
                        q += 1;
                    }
                    if found == outer_depth {
                        out.push(marker_start..q);
                        break;
                    }
                    found += 1;
                } else {
                    // Lazy continuation line — no marker for this depth.
                    break;
                }
            }
            p = line_end + 1;
        }
        out
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

    #[test]
    fn parses_fenced_code_block_with_language() {
        // pulldown reports `range` from the opening fence through the
        // closing fence — the trailing `\n` after the closing fence is
        // outside the construct (it's the same one-trailing-newline
        // behavior paragraphs have, handled in the renderer's empty-
        // paragraph injection).
        let src = "```rust\nfn x() {}\n```\n";
        let nodes = parse(src);
        assert_eq!(nodes.len(), 1);
        match &first(&nodes).kind {
            NodeKind::CodeBlock {
                lang,
                content_range,
                delimiter_ranges,
                info_string_range,
            } => {
                assert_eq!(lang.as_deref(), Some("rust"));
                assert_eq!(&src[content_range.clone()], "fn x() {}\n");
                assert_eq!(delimiter_ranges.len(), 2);
                assert_eq!(&src[delimiter_ranges[0].clone()], "```");
                assert_eq!(&src[delimiter_ranges[1].clone()], "```");
                assert_eq!(
                    info_string_range.as_ref().map(|r| &src[r.clone()]),
                    Some("rust")
                );
            }
            other => panic!("expected fenced code block, got {other:?}"),
        }
    }

    #[test]
    fn parses_fenced_code_block_without_language() {
        let src = "```\nplain\n```";
        let nodes = parse(src);
        match &first(&nodes).kind {
            NodeKind::CodeBlock {
                lang,
                content_range,
                delimiter_ranges,
                info_string_range,
            } => {
                assert_eq!(lang.as_deref(), Some(""));
                assert_eq!(&src[content_range.clone()], "plain\n");
                assert_eq!(&src[delimiter_ranges[0].clone()], "```");
                assert_eq!(&src[delimiter_ranges[1].clone()], "```");
                assert!(info_string_range.is_none());
            }
            other => panic!("expected fenced code block, got {other:?}"),
        }
    }

    #[test]
    fn parses_tilde_fenced_code_block() {
        let src = "~~~js\nlet x = 1;\n~~~\n";
        let nodes = parse(src);
        match &first(&nodes).kind {
            NodeKind::CodeBlock {
                lang,
                content_range,
                delimiter_ranges,
                info_string_range,
            } => {
                assert_eq!(lang.as_deref(), Some("js"));
                assert_eq!(&src[content_range.clone()], "let x = 1;\n");
                assert_eq!(&src[delimiter_ranges[0].clone()], "~~~");
                assert_eq!(&src[delimiter_ranges[1].clone()], "~~~");
                assert_eq!(
                    info_string_range.as_ref().map(|r| &src[r.clone()]),
                    Some("js")
                );
            }
            other => panic!("expected fenced code block, got {other:?}"),
        }
    }

    #[test]
    fn parses_unterminated_fenced_code_block() {
        // CommonMark allows a fenced block with no closing fence — the
        // block extends to end-of-document. Useful for live editing.
        let src = "```rust\nfn x() {}\n";
        let nodes = parse(src);
        match &first(&nodes).kind {
            NodeKind::CodeBlock {
                lang,
                content_range,
                delimiter_ranges,
                info_string_range,
            } => {
                assert_eq!(lang.as_deref(), Some("rust"));
                assert_eq!(&src[content_range.clone()], "fn x() {}\n");
                assert_eq!(delimiter_ranges.len(), 1);
                assert_eq!(&src[delimiter_ranges[0].clone()], "```");
                assert_eq!(
                    info_string_range.as_ref().map(|r| &src[r.clone()]),
                    Some("rust")
                );
            }
            other => panic!("expected fenced code block, got {other:?}"),
        }
    }

    // ---- Blockquotes ----------------------------------------------------

    #[test]
    fn parses_simple_blockquote_with_one_prefix_per_line() {
        let src = "> first\n> second\n";
        let nodes = parse(src);
        assert_eq!(nodes.len(), 1);
        match &first(&nodes).kind {
            NodeKind::BlockQuote { prefix_ranges } => {
                assert_eq!(prefix_ranges.len(), 2);
                assert_eq!(&src[prefix_ranges[0].clone()], "> ");
                assert_eq!(&src[prefix_ranges[1].clone()], "> ");
            }
            other => panic!("expected blockquote, got {other:?}"),
        }
    }

    #[test]
    fn nested_blockquote_records_only_its_own_marker() {
        // `> > deep` — the outer blockquote owns the first `>`, the
        // inner the second. Each node's prefix_ranges list contains
        // exactly one entry, both on the same source line.
        let src = "> > deep\n";
        let nodes = parse(src);
        assert_eq!(nodes.len(), 1);
        let outer = first(&nodes);
        match &outer.kind {
            NodeKind::BlockQuote { prefix_ranges } => {
                assert_eq!(prefix_ranges.len(), 1);
                assert_eq!(&src[prefix_ranges[0].clone()], "> ");
                assert_eq!(prefix_ranges[0], 0..2);
            }
            other => panic!("expected outer blockquote, got {other:?}"),
        }
        let inner = outer
            .children
            .iter()
            .find(|c| matches!(c.kind, NodeKind::BlockQuote { .. }))
            .expect("inner blockquote child");
        match &inner.kind {
            NodeKind::BlockQuote { prefix_ranges } => {
                assert_eq!(prefix_ranges.len(), 1);
                assert_eq!(&src[prefix_ranges[0].clone()], "> ");
                assert_eq!(prefix_ranges[0], 2..4);
            }
            other => panic!("expected inner blockquote, got {other:?}"),
        }
    }

    #[test]
    fn blockquote_with_marker_only_line_records_single_byte_range() {
        // `>` followed by no space is still a valid marker. Range is
        // 1 byte wide because there's no trailing space to consume.
        let src = "> a\n>\n> b\n";
        let nodes = parse(src);
        match &first(&nodes).kind {
            NodeKind::BlockQuote { prefix_ranges } => {
                assert_eq!(prefix_ranges.len(), 3);
                assert_eq!(&src[prefix_ranges[0].clone()], "> ");
                assert_eq!(&src[prefix_ranges[1].clone()], ">");
                assert_eq!(&src[prefix_ranges[2].clone()], "> ");
            }
            other => panic!("expected blockquote, got {other:?}"),
        }
    }

    #[test]
    fn blockquote_around_paragraph_keeps_paragraph_child() {
        let src = "> hi\n";
        let nodes = parse(src);
        let bq = first(&nodes);
        assert!(
            bq.children
                .iter()
                .any(|c| matches!(c.kind, NodeKind::Paragraph)),
            "blockquote must contain its inner paragraph as a child node",
        );
    }

    #[test]
    fn fenced_code_block_with_no_inner_markdown_parsing() {
        // `**bold**` inside a code block is literal — no Strong child
        // node should appear in the parse tree.
        let src = "```\n**not bold**\n```";
        let nodes = parse(src);
        let block = first(&nodes);
        assert!(matches!(block.kind, NodeKind::CodeBlock { .. }));
        let has_strong = block
            .children
            .iter()
            .any(|c| matches!(c.kind, NodeKind::Strong { .. }));
        assert!(!has_strong, "code block must not contain inline parses");
    }
}
