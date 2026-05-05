//! `BlockElement` — a gpui `Element` that paints exactly one `RenderBlock`.
//!
//! One element per block keeps full-width decorations (code-block backgrounds,
//! blockquote borders) trivial when those land later — they become per-block
//! quads instead of a custom layout-fragment subclass. For the current
//! minimum-viable cut (paragraphs + ATX headings only), the only per-block
//! complication is the `display_to_source` map: each shaped line carries a
//! per-display-byte `Vec<usize>` mapping back to source bytes, so cursor
//! position math survives delimiters being literally removed from the
//! shaped string.

use std::ops::Range;
use std::sync::Arc;

use gpui::{
    App, Bounds, Element, ElementId, ElementInputHandler, Entity, FontStyle, FontWeight,
    GlobalElementId, InspectorElementId, IntoElement, LayoutId, Pixels, Point, SharedString, Size,
    StrikethroughStyle, Style, TextRun, Window, WrappedLine, fill, point, px, relative, size,
};
use smallvec::SmallVec;

use crate::editor::MarkdownEditor;
use crate::render_spec::{BlockKind, InlineStyle, RenderBlock};
use crate::state::Selection;
use crate::style::MarkdownStyle;

pub struct BlockElement {
    block: RenderBlock,
    block_index: usize,
    is_last_block: bool,
    editor: Entity<MarkdownEditor>,
    style: MarkdownStyle,
}

impl BlockElement {
    pub fn new(
        block: RenderBlock,
        block_index: usize,
        is_last_block: bool,
        editor: Entity<MarkdownEditor>,
        style: MarkdownStyle,
    ) -> Self {
        Self {
            block,
            block_index,
            is_last_block,
            editor,
            style,
        }
    }
}

impl IntoElement for BlockElement {
    type Element = Self;
    fn into_element(self) -> Self::Element {
        self
    }
}

/// One shaped, possibly soft-wrapped, logical line.
pub struct LaidOutLine {
    pub line: Arc<WrappedLine>,
    pub origin: Point<Pixels>,
    pub row_height: Pixels,
    pub wrapped_height: Pixels,
    /// Source byte range covered by this line (including trailing `\n` if
    /// any, so the cursor at end-of-line resolves to the next paragraph).
    pub source_range: Range<usize>,
    /// `display_to_source[i]` is the source byte index for display byte `i`.
    /// Length == `line.text.len() + 1`; the trailing entry maps EOL to
    /// `source_range.end`.
    pub display_to_source: Vec<usize>,
}

impl LaidOutLine {
    pub fn contains_source_offset(&self, offset: usize) -> bool {
        offset >= self.source_range.start && offset <= self.source_range.end
    }

    fn display_offset_for_source(&self, source_offset: usize) -> usize {
        if source_offset <= self.source_range.start {
            return 0;
        }
        if source_offset >= self.source_range.end {
            return self.line.text.len();
        }
        for (i, &src) in self.display_to_source.iter().enumerate() {
            if src >= source_offset {
                return i;
            }
        }
        self.line.text.len()
    }

    pub fn local_position_for_source_offset(&self, source_offset: usize) -> Point<Pixels> {
        let display = self.display_offset_for_source(source_offset);
        self.line
            .position_for_index(display, self.row_height)
            .unwrap_or_else(|| point(px(0.0), px(0.0)))
    }

    pub fn source_offset_for_local_point(&self, local: Point<Pixels>) -> usize {
        let mut p = local;
        if p.x < px(0.0) {
            p.x = px(0.0);
        }
        if p.y < px(0.0) {
            p.y = px(0.0);
        }
        let display_idx = match self.line.closest_index_for_position(p, self.row_height) {
            Ok(i) => i,
            Err(i) => i,
        };
        if display_idx >= self.display_to_source.len() {
            return self.source_range.end;
        }
        self.display_to_source
            .get(display_idx)
            .copied()
            .unwrap_or(self.source_range.end)
    }

    pub fn row_count(&self) -> usize {
        1 + self.line.wrap_boundaries().len()
    }
}

pub struct LaidOutBlock {
    pub block_bounds: Bounds<Pixels>,
    pub lines: Vec<LaidOutLine>,
    pub source_range: Range<usize>,
}

pub struct PrepaintState {
    laid_out: LaidOutBlock,
    cursor_quad: Option<gpui::PaintQuad>,
    selection_quads: Vec<gpui::PaintQuad>,
}

impl Element for BlockElement {
    type RequestLayoutState = ();
    type PrepaintState = PrepaintState;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        let font_size = font_size_for_block(&self.block.kind, &self.style);
        let line_height = font_size * self.style.line_height.0;
        let spacing_above = spacing_above_for_block(&self.block.kind, &self.style);

        let source = self.editor.read(cx).state.markdown.clone();
        style.size.width = relative(1.0).into();

        let block_clone = self.block.clone();
        let style_clone = self.style.clone();
        let id = window.request_measured_layout(
            style,
            move |known_dimensions, available_space, window, _cx| {
                let avail_w = known_dimensions
                    .width
                    .or(match available_space.width {
                        gpui::AvailableSpace::Definite(w) => Some(w),
                        _ => None,
                    })
                    .unwrap_or(px(f32::INFINITY));
                let wrap_w = avail_w.max(px(1.0));
                let lines = shape_block_lines(
                    &source,
                    &block_clone,
                    &style_clone,
                    font_size,
                    Some(wrap_w),
                    window,
                );
                let mut h = spacing_above;
                if lines.is_empty() {
                    h += line_height;
                }
                for line in &lines {
                    h += line_height * ((line.line.wrap_boundaries().len() as f32) + 1.0);
                }
                Size {
                    width: avail_w,
                    height: h,
                }
            },
        );
        (id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let (selection, source) = {
            let editor = self.editor.read(cx);
            (editor.state.selection, editor.state.markdown.clone())
        };

        let style = self.style.clone();
        let font_size = font_size_for_block(&self.block.kind, &style);
        let line_height = font_size * style.line_height.0;
        let spacing_above = spacing_above_for_block(&self.block.kind, &style);

        let block_top = bounds.origin.y + spacing_above;
        let line_origin_x = bounds.origin.x;
        let block_width = bounds.size.width;

        let wrap_w = Some(block_width.max(px(1.0)));
        let shaped = shape_block_lines(&source, &self.block, &style, font_size, wrap_w, window);

        let mut lines: Vec<LaidOutLine> = Vec::new();
        let mut cursor_y = block_top;
        for sl in shaped {
            let wrapped_h = line_height * ((sl.line.wrap_boundaries().len() as f32) + 1.0);
            let origin = point(line_origin_x, cursor_y);
            lines.push(LaidOutLine {
                line: sl.line,
                origin,
                row_height: line_height,
                wrapped_height: wrapped_h,
                source_range: sl.source_range,
                display_to_source: sl.display_to_source,
            });
            cursor_y += wrapped_h;
        }

        if lines.is_empty() {
            // Empty block — fabricate a zero-content shaped line so cursor
            // positioning still works on truly empty paragraphs.
            if let Some(line) = empty_shaped_line(font_size, window) {
                lines.push(LaidOutLine {
                    line,
                    origin: point(line_origin_x, cursor_y),
                    row_height: line_height,
                    wrapped_height: line_height,
                    source_range: self.block.source_range.clone(),
                    display_to_source: vec![self.block.source_range.start],
                });
                cursor_y += line_height;
            }
        }

        let block_bounds = Bounds::new(
            point(line_origin_x, block_top),
            size(block_width, cursor_y - block_top),
        );

        let laid_out = LaidOutBlock {
            block_bounds,
            lines,
            source_range: self.block.source_range.clone(),
        };

        let (mut cursor_quad, selection_quads) =
            build_caret_and_selection(&laid_out, selection, &style, self.is_last_block);

        if cursor_quad.is_none() && source.is_empty() && self.block_index == 0 {
            // Truly empty document — paint a cursor at the origin so the
            // user sees the editor is focused.
            cursor_quad = Some(fill(
                Bounds::new(bounds.origin, size(px(2.0), line_height)),
                style.caret_color,
            ));
        }

        PrepaintState {
            laid_out,
            cursor_quad,
            selection_quads,
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let (focus_handle, should_register) = self.editor.update(cx, |editor, _| {
            let should = !editor.frame_input_handler_set;
            editor.frame_input_handler_set = true;
            (editor.focus_handle.clone(), should)
        });
        if should_register {
            let editor_bounds = self.editor.read(cx).last_bounds.unwrap_or(bounds);
            window.handle_input(
                &focus_handle,
                ElementInputHandler::new(editor_bounds, self.editor.clone()),
                cx,
            );
        }

        for q in prepaint.selection_quads.drain(..) {
            window.paint_quad(q);
        }

        for laid in &prepaint.laid_out.lines {
            let _ = laid.line.paint(
                laid.origin,
                laid.row_height,
                gpui::TextAlign::Left,
                None,
                window,
                cx,
            );
        }

        if focus_handle.is_focused(window)
            && let Some(quad) = prepaint.cursor_quad.take()
        {
            window.paint_quad(quad);
        }

        let laid_out = std::mem::replace(
            &mut prepaint.laid_out,
            LaidOutBlock {
                block_bounds: Bounds::default(),
                lines: Vec::new(),
                source_range: 0..0,
            },
        );
        let block_index = self.block_index;
        self.editor.update(cx, |editor, _| {
            editor.last_blocks.insert(block_index, laid_out);
            editor.last_bounds = Some(match editor.last_bounds {
                Some(prev) if block_index != 0 => union_bounds(prev, bounds),
                _ => bounds,
            });
        });
    }
}

fn union_bounds(a: Bounds<Pixels>, b: Bounds<Pixels>) -> Bounds<Pixels> {
    let left = a.left().min(b.left());
    let top = a.top().min(b.top());
    let right = a.right().max(b.right());
    let bottom = a.bottom().max(b.bottom());
    Bounds::from_corners(point(left, top), point(right, bottom))
}

fn empty_shaped_line(font_size: Pixels, window: &mut Window) -> Option<Arc<WrappedLine>> {
    window
        .text_system()
        .shape_text(SharedString::from(""), font_size, &[], None, None)
        .ok()
        .and_then(|mut v| v.drain(..).next())
        .map(Arc::new)
}

fn font_size_for_block(kind: &BlockKind, style: &MarkdownStyle) -> Pixels {
    match kind {
        BlockKind::Heading { level } => style.size_for_heading(*level),
        BlockKind::Paragraph => style.font_size,
    }
}

fn spacing_above_for_block(kind: &BlockKind, style: &MarkdownStyle) -> Pixels {
    let rems_factor = match kind {
        BlockKind::Heading { level } if *level <= 2 => 1.5,
        BlockKind::Heading { .. } => 1.25,
        BlockKind::Paragraph => style.paragraph_gap.0,
    };
    px(f32::from(style.font_size) * rems_factor)
}

// ---------- Shaping ----------

struct ShapedLine {
    line: Arc<WrappedLine>,
    source_range: Range<usize>,
    display_to_source: Vec<usize>,
}

/// One block, one logical line at a time. For each line we build the display
/// string (delimiters in `hidden_ranges` are dropped), the per-display-byte
/// `display_to_source` map, and the `TextRun`s, then call
/// `text_system().shape_text` with `wrap_width` so soft-wrap rows are
/// computed.
fn shape_block_lines(
    source: &str,
    block: &RenderBlock,
    style: &MarkdownStyle,
    font_size: Pixels,
    wrap_width: Option<Pixels>,
    window: &mut Window,
) -> Vec<ShapedLine> {
    let block_text = match source.get(block.source_range.clone()) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut cursor = block.source_range.start;

    // Synthetic empty paragraphs from `inject_empty_paragraphs` cover a
    // pair of `\n`s in the pairs model — the whole pair is one visible
    // empty row, not one row per `\n`. Without this short-circuit
    // `split_inclusive('\n')` yields one piece per `\n` and we paint two
    // empty rows where one is intended (the `p1\n\n\n\np2` bug:
    // 1 synthetic block, but 2 visible rows between the paragraphs).
    //
    // Hard breaks (`  \n`, `\\\n`) are content, not pure newlines, so
    // they bypass this branch and fall through to the regular path
    // where the post-loop hard-break post-pass handles them.
    if !block_text.is_empty() && block_text.bytes().all(|b| b == b'\n') {
        if let Some(line) = empty_shaped_line(font_size, window) {
            out.push(ShapedLine {
                line,
                source_range: block.source_range.clone(),
                display_to_source: vec![block.source_range.start],
            });
        }
        return out;
    }

    for raw_line in block_text.split_inclusive('\n') {
        let raw_end = cursor + raw_line.len();
        let trailing_nl = raw_line.ends_with('\n');
        let content_end = if trailing_nl { raw_end - 1 } else { raw_end };
        let logical_source_range = cursor..content_end;
        let line_source_range = cursor..raw_end;

        let (display_text, display_to_source) =
            build_display_line(source, &logical_source_range, block);
        let runs = build_runs_for_line(&display_text, &display_to_source, block, style);

        let shared = SharedString::from(display_text);
        let mut shaped_vec = window
            .text_system()
            .shape_text(shared, font_size, &runs, wrap_width, None)
            .unwrap_or_default();
        let line = if shaped_vec.is_empty() {
            empty_shaped_line(font_size, window)
        } else {
            Some(Arc::new(shaped_vec.remove(0)))
        };
        if let Some(line) = line {
            out.push(ShapedLine {
                line,
                source_range: line_source_range,
                display_to_source,
            });
        }
        cursor = raw_end;
        if !trailing_nl {
            break;
        }
    }

    // If the block ends with a hard break (`  \n` or `\\\n`), that
    // trailing `\n` is in-paragraph content (an explicit line break),
    // not a paragraph terminator. Emit a visible empty trailing line so
    // the cursor at `block.range.end` lands below the content row.
    // (Other blocks ending in `\n` — most importantly the synthetic
    // single-`\n` empty paragraphs from `inject_empty_paragraphs` — are
    // *not* hard breaks and don't need this treatment.)
    if ends_with_hard_break(block_text)
        && let Some(last) = out.last_mut()
    {
        if last.source_range.end > last.source_range.start {
            last.source_range.end -= 1;
        }
        if let Some(empty_line) = empty_shaped_line(font_size, window) {
            out.push(ShapedLine {
                line: empty_line,
                source_range: block.source_range.end..block.source_range.end,
                display_to_source: vec![block.source_range.end],
            });
        }
    }

    out
}

fn ends_with_hard_break(s: &str) -> bool {
    let bytes = s.as_bytes();
    let n = bytes.len();
    if n == 0 || bytes[n - 1] != b'\n' {
        return false;
    }
    if n >= 3 && bytes[n - 2] == b' ' && bytes[n - 3] == b' ' {
        return true;
    }
    if n >= 2 && bytes[n - 2] == b'\\' {
        return true;
    }
    false
}

fn build_display_line(
    source: &str,
    line: &Range<usize>,
    block: &RenderBlock,
) -> (String, Vec<usize>) {
    let mut display = String::new();
    let mut map: Vec<usize> = Vec::new();

    let mut pos = line.start;
    while pos < line.end {
        if let Some(h) = block
            .hidden_ranges
            .iter()
            .find(|r| r.start == pos && r.end <= line.end)
        {
            pos = h.end;
            continue;
        }
        let ch = source[pos..line.end]
            .chars()
            .next()
            .expect("non-empty remainder");
        let ch_len = ch.len_utf8();
        for _ in 0..ch_len {
            map.push(pos);
        }
        display.push(ch);
        pos += ch_len;
    }
    map.push(line.end);
    debug_assert_eq!(map.len(), display.len() + 1);
    (display, map)
}

fn build_runs_for_line(
    display_text: &str,
    display_to_source: &[usize],
    block: &RenderBlock,
    style: &MarkdownStyle,
) -> SmallVec<[TextRun; 8]> {
    let mut runs: SmallVec<[TextRun; 8]> = SmallVec::new();
    if display_text.is_empty() {
        return runs;
    }

    let base_font = base_font_for_block(&block.kind, style);
    let base_color = style.text_color;
    let base_weight = base_weight_for_block(&block.kind, style);

    let len = display_text.len();
    let mut i = 0usize;
    while i < len {
        let here_src = display_to_source[i];
        let here_style = effective_inline_style(here_src, block);
        let mut j = i + 1;
        while j < len {
            let next_src = display_to_source[j];
            if effective_inline_style(next_src, block) != here_style {
                break;
            }
            j += 1;
        }

        let merged = here_style;
        let mut run_font = base_font.clone();
        if merged.bold || base_weight == FontWeight::BOLD {
            run_font.weight = FontWeight::BOLD;
        } else if base_weight != FontWeight::NORMAL {
            run_font.weight = base_weight;
        }
        if merged.italic {
            run_font.style = FontStyle::Italic;
        }

        let color = if merged.dimmed {
            style.delimiter_color
        } else {
            base_color
        };

        let strikethrough = if merged.strikethrough {
            Some(StrikethroughStyle {
                thickness: px(1.0),
                color: Some(color),
            })
        } else {
            None
        };

        runs.push(TextRun {
            len: j - i,
            font: run_font,
            color,
            background_color: None,
            underline: None,
            strikethrough,
        });
        i = j;
    }

    runs
}

fn effective_inline_style(src_offset: usize, block: &RenderBlock) -> InlineStyle {
    let mut acc = InlineStyle::default();
    for run in &block.inlines {
        if src_offset >= run.source_range.start && src_offset < run.source_range.end {
            acc = acc.merge(run.style.clone());
        }
    }
    acc
}

fn base_font_for_block(_kind: &BlockKind, style: &MarkdownStyle) -> gpui::Font {
    gpui::Font {
        family: style.font_family.clone(),
        features: gpui::FontFeatures::default(),
        fallbacks: None,
        weight: FontWeight::NORMAL,
        style: FontStyle::Normal,
    }
}

fn base_weight_for_block(kind: &BlockKind, style: &MarkdownStyle) -> FontWeight {
    let level = match kind {
        BlockKind::Heading { level } => Some(*level),
        _ => None,
    };
    match level {
        Some(l) if style.heading_is_bold(l) => FontWeight::BOLD,
        Some(_) => FontWeight::SEMIBOLD,
        None => FontWeight::NORMAL,
    }
}

// ---------- Cursor / selection ----------

fn build_caret_and_selection(
    block: &LaidOutBlock,
    selection: Selection,
    style: &MarkdownStyle,
    is_last_block: bool,
) -> (Option<gpui::PaintQuad>, Vec<gpui::PaintQuad>) {
    let cursor_offset = selection.head();
    let cursor_color = style.caret_color;
    let selection_color = style.selection_color;

    let mut cursor: Option<gpui::PaintQuad> = None;
    let mut boundary_fallback: Option<gpui::PaintQuad> = None;
    let mut sel_quads: Vec<gpui::PaintQuad> = Vec::new();

    let sel_range = selection.selection_range();
    let has_selection = sel_range.start != sel_range.end;

    let cursor_in_block =
        cursor_offset >= block.source_range.start && cursor_offset <= block.source_range.end;

    for line in &block.lines {
        let lo = line.source_range.start;
        let hi = line.source_range.end;
        if cursor_in_block && cursor.is_none() {
            let strict = cursor_offset >= lo && cursor_offset < hi;
            let boundary = cursor_offset == hi && (is_last_block || hi == block.source_range.end);
            if strict || (boundary && boundary_fallback.is_none()) {
                let local = line.local_position_for_source_offset(cursor_offset);
                let x = line.origin.x + local.x;
                let y = line.origin.y + local.y;
                let quad = fill(
                    Bounds::new(point(x, y), size(px(2.0), line.row_height)),
                    cursor_color,
                );
                if strict {
                    cursor = Some(quad);
                } else {
                    boundary_fallback = Some(quad);
                }
            }
        }
        if has_selection {
            let lo_clamped = sel_range.start.max(lo);
            let hi_clamped = sel_range.end.min(hi);
            if hi_clamped > lo_clamped {
                paint_selection_for_line(
                    line,
                    lo_clamped,
                    hi_clamped,
                    hi,
                    selection_color,
                    &mut sel_quads,
                );
            }
        }
    }

    (cursor.or(boundary_fallback), sel_quads)
}

fn paint_selection_for_line(
    line: &LaidOutLine,
    lo: usize,
    hi: usize,
    line_hi: usize,
    color: gpui::Hsla,
    out: &mut Vec<gpui::PaintQuad>,
) {
    let start = line.local_position_for_source_offset(lo);
    let end = line.local_position_for_source_offset(hi);
    let row_height = line.row_height;
    let eol_pad = if hi == line_hi { px(6.0) } else { px(0.0) };

    if start.y == end.y {
        let x0 = line.origin.x + start.x;
        let x1 = line.origin.x + end.x + eol_pad;
        let y0 = line.origin.y + start.y;
        out.push(fill(
            Bounds::from_corners(point(x0, y0), point(x1, y0 + row_height)),
            color,
        ));
        return;
    }

    let row_count = line.row_count();
    let line_width = line.line.width();
    let start_row = (f32::from(start.y) / f32::from(row_height)).round() as usize;
    let end_row = (f32::from(end.y) / f32::from(row_height)).round() as usize;

    let y_start = line.origin.y + start.y;
    out.push(fill(
        Bounds::from_corners(
            point(line.origin.x + start.x, y_start),
            point(line.origin.x + line_width, y_start + row_height),
        ),
        color,
    ));

    for row in (start_row + 1)..end_row.min(row_count) {
        let y = line.origin.y + row_height * (row as f32);
        out.push(fill(
            Bounds::from_corners(
                point(line.origin.x, y),
                point(line.origin.x + line_width, y + row_height),
            ),
            color,
        ));
    }

    let y_end = line.origin.y + end.y;
    out.push(fill(
        Bounds::from_corners(
            point(line.origin.x, y_end),
            point(line.origin.x + end.x + eol_pad, y_end + row_height),
        ),
        color,
    ));
}

// ---------------------------------------------------------------------------
// Whitespace shaping tests — companion to the block-count tests in
// `render.rs`. Those check `inject_empty_paragraphs` produces the right
// number of `RenderBlock`s; these check `shape_block_lines` produces the
// right number of *visible* shaped lines per block. Both invariants must
// hold for the rendered editor to match user intent — the bug that
// motivated this module split was a `\n\n` synthetic block (1 block per
// the renderer) shaping into 2 visible empty rows.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::render::render;
    use crate::state::{EditorState, Selection};
    use gpui::{AppContext, Context, Render, TestAppContext, WindowOptions};

    struct EmptyRoot;
    impl Render for EmptyRoot {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            gpui::div()
        }
    }

    /// Render `src` end-to-end, then shape each block through
    /// `shape_block_lines` and return the visible-line count per block.
    /// `wrap_w` is wide enough that no soft-wrap kicks in for the inputs
    /// we test.
    fn shape_visible_row_counts(cx: &mut TestAppContext, src: &str) -> Vec<usize> {
        let state = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(0),
        };
        let tree = parse(src);
        let blocks = render(&state, &tree).blocks;
        let src_owned = src.to_string();

        let handle = cx.update(|cx| {
            gpui_component::init(cx);
            cx.open_window(WindowOptions::default(), |_, cx| cx.new(|_| EmptyRoot))
                .expect("open window")
        });

        cx.update_window(handle.into(), |_, window, cx| {
            let style = MarkdownStyle::from_theme(cx);
            blocks
                .iter()
                .map(|b| {
                    let font_size = font_size_for_block(&b.kind, &style);
                    shape_block_lines(&src_owned, b, &style, font_size, Some(px(720.0)), window)
                        .len()
                })
                .collect()
        })
        .expect("update window")
    }

    #[gpui::test]
    fn paragraph_break_alone_shapes_one_line_per_paragraph(cx: &mut TestAppContext) {
        // `p1\n\np2` — two real paragraphs, no synthetic empties between.
        // Each paragraph trims to its content (no trailing `\n`) so one
        // visible row each.
        let counts = shape_visible_row_counts(cx, "p1\n\np2");
        assert_eq!(counts, vec![1, 1]);
    }

    #[gpui::test]
    fn extra_inter_block_pair_shapes_one_visible_empty_row(cx: &mut TestAppContext) {
        // `p1\n\n\n\np2` — 1 synthetic empty between (range 3..5,
        // bytes "\n\n"). The pairs model says one visible empty row per
        // synthetic, not two.
        let counts = shape_visible_row_counts(cx, "p1\n\n\n\np2");
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[gpui::test]
    fn six_newlines_between_real_blocks_shape_two_visible_empty_rows(cx: &mut TestAppContext) {
        // The user-readable form: paragraph 1, two visible empty rows,
        // paragraph 2. In the pairs model that's 6 `\n`s = paragraph
        // break + 2 empty pairs = 2 synthetic blocks shaping to 1 line
        // each.
        let counts = shape_visible_row_counts(cx, "paragraph 1\n\n\n\n\n\nparagraph 2");
        assert_eq!(counts, vec![1, 1, 1, 1]);
    }

    #[gpui::test]
    fn trailing_pair_shapes_one_visible_empty_row(cx: &mut TestAppContext) {
        // Enter at end of a paragraph: one trailing pair, one visible
        // trailing empty row.
        let counts = shape_visible_row_counts(cx, "paragraph 1\n\n");
        assert_eq!(counts, vec![1, 1]);
    }

    #[gpui::test]
    fn three_trailing_pairs_shape_three_visible_empty_rows(cx: &mut TestAppContext) {
        // Three Enters at the end → three trailing visible empty rows.
        let counts = shape_visible_row_counts(cx, "ab\n\n\n\n\n\n");
        assert_eq!(counts, vec![1, 1, 1, 1]);
    }

    #[gpui::test]
    fn leading_pair_shapes_one_visible_empty_row_above(cx: &mut TestAppContext) {
        let counts = shape_visible_row_counts(cx, "\n\np1");
        assert_eq!(counts, vec![1, 1]);
    }

    #[gpui::test]
    fn trailing_hard_break_shapes_two_lines_in_one_block(cx: &mut TestAppContext) {
        // `paragraph 1  \n` is one paragraph with a trailing hard break:
        // one block, two shaped lines (the content row + the empty
        // trailing row inside the same paragraph). Crucially this
        // bypasses the new all-newlines short-circuit because the block
        // text contains spaces.
        let counts = shape_visible_row_counts(cx, "paragraph 1  \n");
        assert_eq!(counts, vec![2]);
    }

    #[gpui::test]
    fn empty_doc_shapes_zero_lines_at_shape_layer(cx: &mut TestAppContext) {
        // The single anchor block for "" has range 0..0 — block_text is
        // "", split_inclusive yields nothing, no shaped lines. The
        // visible empty row is fabricated by `prepaint`'s fallback, not
        // by `shape_block_lines`. This test pins that contract: the
        // shape layer doesn't pretend "" has a row.
        let counts = shape_visible_row_counts(cx, "");
        assert_eq!(counts, vec![0]);
    }

    #[gpui::test]
    fn heading_with_inter_block_empties_shapes_one_per_synthetic(cx: &mut TestAppContext) {
        // Headings use the same inter-block formula as paragraphs. Two
        // synthetic empties between heading and body should shape to one
        // line each.
        let counts = shape_visible_row_counts(cx, "# title\n\n\n\n\n\nbody");
        assert_eq!(counts, vec![1, 1, 1, 1]);
    }
}
