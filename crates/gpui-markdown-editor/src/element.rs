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
    App, BorderStyle, Bounds, ContentMask, Edges, Element, ElementId, ElementInputHandler, Entity,
    FontStyle, FontWeight, GlobalElementId, InspectorElementId, IntoElement, LayoutId, Pixels,
    Point, ScrollDelta, ScrollWheelEvent, SharedString, Size, StrikethroughStyle, Style, TextRun,
    Window, WrappedLine, fill, point, px, quad, relative, size,
};
use smallvec::SmallVec;

use crate::editor::MarkdownEditorState;
use crate::render_spec::{
    BlockKind, Container, ImageOverlay, InlineRun, InlineStyle, ListItemKind, MathOverlay,
    RenderBlock, Substitution,
};
use crate::state::Selection;
use crate::style::MarkdownStyle;

pub struct BlockElement {
    block: RenderBlock,
    block_index: usize,
    is_last_block: bool,
    /// Source-range start of the block immediately after this one in
    /// document order, if any. Used so the cursor at this block's
    /// `source_range.end` only renders here when the next block doesn't
    /// strictly contain that offset — otherwise the caret would paint
    /// twice (once at this block's end-of-line, once at the next
    /// block's start). Adjacent blocks share boundaries in the
    /// trailing/leading empty-paragraph cases, where `inject_empty_paragraphs`
    /// puts the empty's start at the previous block's end.
    next_block_start: Option<usize>,
    /// Container chain of the previous block in document order, if
    /// any. `None` at doc start. Used to detect "container boundary"
    /// transitions — blockquote-to-paragraph and friends — and add
    /// extra breathing room above this block when the chains differ.
    prev_containers: Option<Vec<Container>>,
    /// Symmetric to `prev_containers` for the block immediately
    /// after this one. `None` at doc end.
    next_containers: Option<Vec<Container>>,
    editor: Entity<MarkdownEditorState>,
    style: MarkdownStyle,
}

impl BlockElement {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block: RenderBlock,
        block_index: usize,
        is_last_block: bool,
        next_block_start: Option<usize>,
        prev_containers: Option<Vec<Container>>,
        next_containers: Option<Vec<Container>>,
        editor: Entity<MarkdownEditorState>,
        style: MarkdownStyle,
    ) -> Self {
        Self {
            block,
            block_index,
            is_last_block,
            next_block_start,
            prev_containers,
            next_containers,
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
    /// True for code-block fence lines — the opening and closing
    /// fences. Distinguishes them from content lines so paint can:
    ///
    /// * keep fence lines pinned (they don't translate with
    ///   `code_block.scroll_x`);
    /// * paint fences *outside* the content-area `with_content_mask`
    ///   so they aren't clipped when the user scrolls horizontally;
    /// * always reserve a row of vertical space for them, even when
    ///   they're hidden (cursor outside the block) — without this
    ///   the block would shrink/grow vertically as it gains/loses
    ///   focus.
    pub is_delimiter: bool,
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
        if let Some(p) = self.line.position_for_index(display, self.row_height) {
            return p;
        }
        // Fallback for end-of-line cursors when the shaped text
        // ended in whitespace that gpui collapsed at a soft-break
        // boundary. `display` sits at `text.len()` but is past the
        // last *glyph* in the wrap layout, so `position_for_index`
        // walks all wrap rows and returns `None`. Place the caret
        // at the line's right edge on its last row instead — that's
        // where the user expects the typing position after a
        // trailing space. Without this, the caret quad lands at
        // `(0, 0)` and visually disappears against the leftmost
        // glyph of the line.
        let last_row = self.line.wrap_boundaries().len();
        let y = self.row_height * (last_row as f32);
        point(self.line.width(), y)
    }

    /// Like [`Self::local_position_for_source_offset`], but when `downstream` is
    /// true and `source_offset` sits exactly on a soft-wrap boundary (the start
    /// of a wrap row after the first), returns that lower row's LEFT edge
    /// `(0, row*h)` instead of the upper row's end. gpui's `position_for_index`
    /// always resolves a boundary offset to the end of the upper row (upper
    /// affinity); this biases it to the lower row's start so a caret placed by
    /// Down/Home/Right renders — and vertical motion computes its row — where the
    /// user expects. Non-boundary offsets are unaffected.
    pub fn local_position_for_source_offset_biased(
        &self,
        source_offset: usize,
        downstream: bool,
    ) -> Point<Pixels> {
        let upper = self.local_position_for_source_offset(source_offset);
        if !downstream {
            return upper;
        }
        let row_h = self.row_height;
        if row_h <= px(0.) {
            return upper;
        }
        let k = (upper.y / row_h).round() as i64; // upper-affinity row index
        let last_row = self.line.wrap_boundaries().len() as i64; // rows are 0..=last_row
        if k < 0 || k >= last_row {
            return upper; // already on the last row
        }
        // Is `source_offset` exactly the boundary that STARTS row k+1? Probe the
        // VERTICAL CENTER of row k+1 (`(k + 1.5) * row_h`), never its top edge:
        // gpui's `closest_index_for_position` floors `y / line_height`, and a
        // top-edge sample `(k + 1) * row_h` divided back by `row_h` float-rounds
        // to `k + 0.999… → k` (the row above), so a top-edge probe would ask the
        // wrong row. The half-row offset keeps the floor exact.
        let display = self.display_offset_for_source(source_offset);
        let next_row_start = match self
            .line
            .closest_index_for_position(point(px(0.0), row_h * ((k + 1) as f32 + 0.5)), row_h)
        {
            Ok(i) | Err(i) => i,
        };
        if next_row_start == display {
            point(px(0.0), row_h * ((k + 1) as f32))
        } else {
            upper
        }
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
    /// `Some(ordinal)` when this block is a rendered `BlockKind::Embed` —
    /// recorded **at render time** so the mouse hit-test resolves a click on
    /// the embed container directly, instead of re-deriving embed ranges
    /// from source and matching them against block ranges by equality. That
    /// equality happens to hold today only because `inject_empty_paragraphs`
    /// strips the folded trailing `\n` the same way `embed_blocks` trims it
    /// — an incidental cross-module invariant no test owned; recording the
    /// ordinal at render removes the coupling (and the per-click parse).
    pub embed_ordinal: Option<u64>,
    /// `Some` for a rendered embed: what its quote container actually painted
    /// this frame. Moved out of the paint state once painting is done rather
    /// than cloned, so recording it costs one `Vec` per embed per frame.
    ///
    /// The pieces are display-only — no caret, no hit-testing — so nothing in
    /// the interaction path reads this; it exists so the embed-fidelity
    /// regressions can assert on real laid-out chrome (a list's bullet
    /// glyphs, a code block's panel) instead of eyeballing a snapshot.
    pub embed_content: Option<EmbedContentGeometry>,
}

/// Laid-out content of one rendered embed block, in window coordinates.
/// See [`LaidOutBlock::embed_content`].
#[derive(Default)]
pub struct EmbedContentGeometry {
    /// Every painted text piece: shaped text plus its origin. Includes the
    /// list-marker glyphs (`• `, `1. `, `☑ `), which are chrome rather than
    /// document text and so appear in no shaped line.
    pub pieces: Vec<(SharedString, Point<Pixels>)>,
    /// Rounded background panels behind embedded fenced code blocks.
    pub code_panels: Vec<Bounds<Pixels>>,
    /// Origins of the typeset math (inline and display) painted inside.
    pub math_origins: Vec<Point<Pixels>>,
    /// Hairline rules — embedded tables' row boundaries and thematic breaks.
    pub rules: Vec<Bounds<Pixels>>,
    /// Border bars of blockquotes nested inside the embedded content.
    pub bars: Vec<Bounds<Pixels>>,
}

pub struct PrepaintState {
    laid_out: LaidOutBlock,
    cursor_quad: Option<TaggedQuad>,
    selection_quads: Vec<TaggedQuad>,
    /// Wash quads for host-supplied highlight ranges (see
    /// [`crate::highlight`]) — painted before the selection quads so a
    /// selection over highlighted text reads on top of the wash.
    highlight_quads: Vec<TaggedQuad>,
    /// `Some` only for code blocks. Holds the geometry needed during
    /// paint to draw the rounded background, clip content to the
    /// visible band, and overlay the horizontal scrollbar.
    code_block_paint: Option<CodeBlockPaint>,
    /// `Some` only for `BlockKind::Table`. Horizontal-scroll geometry
    /// (tables never soft-wrap) plus display mode's hairline rules.
    table_paint: Option<TablePaint>,
    /// `Some` only for `BlockKind::DisplayMath` in *display* mode
    /// (cursor outside the math). Holds the typeset math + paint
    /// origin. Edit mode falls back to text shaping and uses the
    /// regular `laid_out.lines` path.
    math_paint: Option<MathPaint>,
    /// Inline-math overlays that were typeset and substituted into
    /// the shaped lines. Each entry carries the typeset
    /// `MathLayout` plus the source range that was substituted —
    /// the paint phase locates the substitution's display offset in
    /// `laid_out.lines` and paints the math there.
    inline_math: Vec<InlineMathPaintSpec>,
    /// `Some` only for `BlockKind::Image` in *display* mode (cursor
    /// outside the image). Holds the loaded image + paint bounds.
    /// Edit mode falls back to text shaping. Loading / failure
    /// states are absorbed here too: `Loading` reserves a
    /// placeholder height, `Failed` falls through to the regular
    /// shaped path which renders the source as raw bytes.
    image_paint: Option<ImagePaint>,
    /// Inline-image overlays that were loaded and substituted into
    /// the shaped lines. Same structure as [`inline_math`] — each
    /// entry's `source_range` lets the paint phase locate the
    /// substitution's display offset and paint the image there.
    inline_images: Vec<InlineImagePaintSpec>,
    /// `Some` only for `BlockKind::Embed`. The laid-out mapped markdown
    /// plus the quote chrome (see [`EmbedPaint`]). The regular shaped
    /// path contributes only the fully-hidden marker line (caret /
    /// hit-test anchor); this is what actually paints.
    embed_paint: Option<EmbedPaint>,
}

/// Paint state for a typeset display-math block.
struct MathPaint {
    layout: crate::math::MathLayout,
    origin: Point<Pixels>,
    em_px: Pixels,
}

/// Paint state for a sole-image (`BlockKind::Image`) block in
/// display mode. Mirrors [`MathPaint`] — the image's bounds are
/// pre-computed in `prepaint` and consumed by `paint`.
struct ImagePaint {
    image: std::sync::Arc<gpui::RenderImage>,
    bounds: Bounds<Pixels>,
}

/// One inline-math overlay that's been typeset and is awaiting
/// paint. `source_range` matches the overlay we substituted into
/// the shaped line, so the paint phase can find the substitution's
/// display offset by walking each shaped line's `display_to_source`.
struct InlineMathPaintSpec {
    source_range: Range<usize>,
    layout: crate::math::MathLayout,
    em_px: Pixels,
    display_style: bool,
}

/// One inline-image overlay that's been resolved and is awaiting
/// paint. Mirrors [`InlineMathPaintSpec`] — `source_range` locates
/// the substitution; `display_size` is the image's painted size
/// (already aspect-corrected and height-capped by [`crate::image::inline_size`]);
/// `image` is the `Some` arm of [`crate::image::LoadedImage::Loaded`].
struct InlineImagePaintSpec {
    source_range: Range<usize>,
    image: std::sync::Arc<gpui::RenderImage>,
    display_size: Size<Pixels>,
}

/// Padding character for math substitutions. U+202F NARROW NO-BREAK
/// SPACE: ~⅙ em wide in fonts that include it, non-breaking so a
/// pad-run never wraps mid-construct. Most modern body fonts ship
/// this glyph; fonts that don't fall back to whatever the platform
/// substitutes (typically a regular-width space) — `measure_pad_advance`
/// reads the actual shaped advance, so the math is sized to whatever
/// the font produces, just with worse precision in the fallback case.
const MATH_PAD_CHAR: char = '\u{202F}';

/// Vertical reservation a single shaped line needs *beyond* the
/// body line height to fit the tallest inline-math overlay sitting
/// on it. `top` extends the row above its natural top edge; `bottom`
/// extends past its natural bottom. Both are zero for math-free
/// lines and for math whose ink overshoot fits inside the body
/// font's natural leading. `gpui::WrappedLine::paint` centers text
/// within the row_height we pass it, so a math row whose math fits
/// inside the existing body leading needs no extra reservation —
/// the math sits in the leading whitespace gpui was going to leave
/// blank anyway.
#[derive(Debug, Clone, Copy, Default)]
struct MathRowExtra {
    top: Pixels,
    bottom: Pixels,
}

impl MathRowExtra {
    fn total(&self) -> Pixels {
        self.top + self.bottom
    }
}

/// For a given shaped line, find the math overshoot above body
/// ascent and below body descent among any overlays on the line,
/// and convert each to a row *extra* — the overshoot that doesn't
/// fit inside the body's natural half-leading. The layout pads the
/// row by this extra; gpui's centered baseline still lands where
/// it would for a body-only row, so math and surrounding text
/// share a baseline regardless of whether the row was extended.
fn compute_math_row_extra(
    line_source_range: &Range<usize>,
    inline_math: &[InlineMathPaintSpec],
    body_ascent: Pixels,
    body_descent: Pixels,
    line_height: Pixels,
) -> MathRowExtra {
    let mut max_ascent_overshoot = px(0.0);
    let mut max_descent_overshoot = px(0.0);
    for spec in inline_math {
        // Math overlays never straddle a logical line boundary
        // (`augment_block_with_math` substitutes the entire
        // construct with a non-breaking pad run), so the overlay
        // belongs to whichever line contains its `source_range.start`.
        if spec.source_range.start < line_source_range.start
            || spec.source_range.start >= line_source_range.end
        {
            continue;
        }
        let math_size = spec.layout.size(spec.em_px);
        let math_baseline = spec.layout.baseline(spec.em_px);
        let math_descent = math_size.height - math_baseline;
        let ascent_over = (math_baseline - body_ascent).max(px(0.0));
        let descent_over = (math_descent - body_descent).max(px(0.0));
        if ascent_over > max_ascent_overshoot {
            max_ascent_overshoot = ascent_over;
        }
        if descent_over > max_descent_overshoot {
            max_descent_overshoot = descent_over;
        }
    }
    // Body's half-leading — half the gap between the line's natural
    // text height (ascent+descent) and the line_height it's painted
    // in. Math whose overshoot fits within this half-leading needs
    // no extra row reservation; gpui's centering already leaves
    // that much blank space above and below the text glyphs.
    let body_leading_half = ((line_height - body_ascent - body_descent) / 2.0).max(px(0.0));
    MathRowExtra {
        top: (max_ascent_overshoot - body_leading_half).max(px(0.0)),
        bottom: (max_descent_overshoot - body_leading_half).max(px(0.0)),
    }
}

/// Body-font ascent / descent at the given font size for the
/// block's primary font. Used to compute math-row padding —
/// any math overlay taller than the body font extends the row
/// above and/or below.
fn body_metrics_for_block(
    kind: &BlockKind,
    style: &MarkdownStyle,
    em_px: Pixels,
    window: &mut Window,
) -> (Pixels, Pixels) {
    let body_font = base_font_for_block(kind, style);
    let body_font_id = window.text_system().resolve_font(&body_font);
    let ascent = window.text_system().ascent(body_font_id, em_px);
    let descent = window.text_system().descent(body_font_id, em_px);
    (ascent, descent)
}

/// Pre-pass that turns each `MathOverlay` on a block into a
/// width-matched `Substitution` of narrow non-breaking spaces and
/// produces a paint spec the element layer paints atop the shaped
/// line.
///
/// The substitution mechanism is what reserves horizontal space for
/// the typeset math: `build_display_line` replaces the math's
/// source bytes with a run of [`MATH_PAD_CHAR`]s whose total advance
/// approximates the math's pixel width, so surrounding text shapes
/// around it instead of overlapping. The math paints on top of
/// those padding glyphs at their shaped-line position. Narrow no-
/// break spaces minimize the per-construct rounding-up slack to ~1
/// pad-glyph advance (~1 px in body text), and their non-breaking
/// classification keeps line-wrap from splitting a math construct
/// across lines.
///
/// Returns a clone of `block` with the math substitutions appended,
/// plus one `InlineMathPaintSpec` per successfully-typeset overlay.
/// Overlays whose LaTeX fails to parse fall back to dim/mono
/// shaping of the raw source — see [`push_failed_overlay_fallback`].
fn augment_block_with_math(
    block: &RenderBlock,
    source: &str,
    em_px: Pixels,
    style: &MarkdownStyle,
    window: &mut Window,
) -> (RenderBlock, Vec<InlineMathPaintSpec>) {
    if block.math_overlays.is_empty() {
        return (block.clone(), Vec::new());
    }
    // Idempotent — the OnceLock guard keeps subsequent calls cheap.
    // Hosts may also have called this already at app init.
    let _ = crate::math::register_katex_fonts(&window.text_system().clone());

    let pad_advance = measure_pad_advance(em_px, style, &block.kind, window);

    let mut augmented = block.clone();
    let mut paint_specs: Vec<InlineMathPaintSpec> = Vec::new();

    for overlay in &block.math_overlays {
        let Some(latex) = source.get(overlay.content_range.clone()) else {
            // Source bounds got out from under us — fall back to
            // raw shaping (no substitution, no overlay). Treat it
            // the same as a typeset failure so the user at least
            // sees the construct.
            push_failed_overlay_fallback(&mut augmented, overlay);
            continue;
        };
        let mode = if overlay.display_style {
            crate::math::MathMode::Display
        } else {
            crate::math::MathMode::Inline
        };
        let sanitized_latex = sanitize_latex(latex, &block.containers);
        let math_layout = match crate::math::typeset(&sanitized_latex, mode) {
            Ok(l) => l,
            Err(_) => {
                // Typeset failed — render the raw `$..$` source
                // dimmed (delimiters) + mono (content) so the user
                // can see and correct the bad LaTeX. This is the
                // same visual treatment used when the cursor sits
                // strictly inside a (well-formed) math construct,
                // so the failure mode reads as "still markdown,
                // just hasn't been typeset yet."
                push_failed_overlay_fallback(&mut augmented, overlay);
                continue;
            }
        };
        let math_width = math_layout.size(em_px).width;
        // Round up so the math has a tiny bit of trailing slack
        // rather than visually overlapping the next text glyph.
        // Floor would risk over-cropping. Min 1 pad glyph
        // guarantees at least one substitution character lands in
        // the line so `display_to_source` carries the math's
        // source position.
        let pad_count =
            ((f32::from(math_width) / f32::from(pad_advance).max(1.0)).ceil() as usize).max(1);
        let display: String = MATH_PAD_CHAR.to_string().repeat(pad_count);
        augmented.substitutions.push(Substitution {
            source_range: overlay.source_range.clone(),
            display,
        });
        paint_specs.push(InlineMathPaintSpec {
            source_range: overlay.source_range.clone(),
            layout: math_layout,
            em_px,
            display_style: overlay.display_style,
        });
    }

    let _ = (style,); // kept on the signature for future per-overlay font selection
    (augmented, paint_specs)
}

/// Pre-pass that turns each [`ImageOverlay`] on a block into a
/// width-matched [`Substitution`] of narrow non-breaking spaces and
/// produces a paint spec the element layer paints atop the shaped
/// line. Mirrors [`augment_block_with_math`] — the substitution
/// mechanism is identical, only the size source differs (image
/// natural size from gpui's cache vs math typesetter dimensions).
///
/// **`loaded_for_overlay` must be pre-resolved by the caller** —
/// `crate::image::load` ultimately calls `window.use_asset`, which
/// requires the currently-rendering view to be on
/// `rendered_entity_stack`. The measure callback for
/// `Window::request_measured_layout` runs *after* `request_layout`
/// returns and the view is no longer on the stack, so calling
/// `image::load` directly from there panics. Instead, callers fetch
/// once in `request_layout`'s body (or `prepaint`, both of which
/// run with the view on the stack), then pass the resolved states
/// through.
///
/// Loading / failure model parallels math typeset outcomes.
/// `Loaded` → measure via [`crate::image::inline_size`], substitute a
/// width-matched pad run, push a paint spec. `Loading` → reserve a
/// placeholder square (the same height cap), substitute a same-width
/// pad run, skip the paint spec; the asset cache invalidates the view
/// when the load resolves so the next frame's measure sees real
/// dimensions. `Failed` → push a fallback inline run pair (dim
/// delimiters plus visible alt text on the raw source bytes) so the
/// user sees the construct and what the broken target is. No
/// substitution.
fn augment_block_with_images(
    block: &RenderBlock,
    loaded_for_overlay: &[crate::image::LoadedImage],
    em_px: Pixels,
    line_height: Pixels,
    style: &MarkdownStyle,
    window: &mut Window,
) -> (RenderBlock, Vec<InlineImagePaintSpec>) {
    if block.image_overlays.is_empty() {
        return (block.clone(), Vec::new());
    }
    let pad_advance = measure_pad_advance(em_px, style, &block.kind, window);

    let mut augmented = block.clone();
    let mut paint_specs: Vec<InlineImagePaintSpec> = Vec::new();

    for (idx, overlay) in block.image_overlays.iter().enumerate() {
        let loaded = loaded_for_overlay
            .get(idx)
            .cloned()
            .unwrap_or(crate::image::LoadedImage::Loading);
        let (display_size, paint_image) = match loaded {
            crate::image::LoadedImage::Loaded(img) => {
                let size = crate::image::inline_size(
                    img.as_ref(),
                    line_height,
                    crate::image::INLINE_HEIGHT_FACTOR,
                );
                (size, Some(img))
            }
            crate::image::LoadedImage::Loading => {
                (crate::image::inline_placeholder_size(line_height), None)
            }
            crate::image::LoadedImage::Failed => {
                push_failed_image_overlay_fallback(&mut augmented, overlay);
                continue;
            }
        };
        let width = display_size.width;
        if width <= px(0.0) {
            // Degenerate (zero-size image) — treat as failure so the
            // user still sees their alt text and can fix the URL.
            push_failed_image_overlay_fallback(&mut augmented, overlay);
            continue;
        }
        let pad_count =
            ((f32::from(width) / f32::from(pad_advance).max(1.0)).ceil() as usize).max(1);
        let display: String = MATH_PAD_CHAR.to_string().repeat(pad_count);
        augmented.substitutions.push(Substitution {
            source_range: overlay.source_range.clone(),
            display,
        });
        if let Some(image) = paint_image {
            paint_specs.push(InlineImagePaintSpec {
                source_range: overlay.source_range.clone(),
                image,
                display_size,
            });
        }
    }

    (augmented, paint_specs)
}

/// Append fallback inline runs for an image overlay whose load
/// failed: dim the `![` and `](url)` delimiters and shape the alt
/// text in the normal text color so the user sees what they typed
/// and can correct the URL. Mirrors the math typeset-failure
/// fallback. Equivalent visually to the cursor-inside branch in
/// `render::walk_inline`.
fn push_failed_image_overlay_fallback(block: &mut RenderBlock, overlay: &ImageOverlay) {
    let opener = overlay.source_range.start..overlay.alt_range.start;
    let closer = overlay.alt_range.end..overlay.source_range.end;
    for delim in [opener, closer] {
        if !delim.is_empty() {
            block.inlines.push(InlineRun {
                source_range: delim,
                style: InlineStyle::dimmed(),
            });
        }
    }
    // Alt text shapes in the default style — it's user-authored
    // prose, not code, so no mono/code styling here. The default
    // `InlineStyle` (no flags) renders as the normal body color.
}

/// Vertical row extra contributed by inline image overlays on a
/// shaped line. Images are vertically centered on the row (CSS
/// `vertical-align: middle` style), so an image taller than the row
/// extends it symmetrically above and below by half the overshoot.
/// Mirrors [`compute_math_row_extra`] in shape — both return
/// [`MathRowExtra`] so callers can `max`-merge them.
fn compute_image_row_extra(
    line_source_range: &Range<usize>,
    inline_images: &[InlineImagePaintSpec],
    line_height: Pixels,
) -> MathRowExtra {
    let mut max_overshoot = px(0.0);
    for spec in inline_images {
        if spec.source_range.start < line_source_range.start
            || spec.source_range.start >= line_source_range.end
        {
            continue;
        }
        if spec.display_size.height > line_height {
            let overshoot = (spec.display_size.height - line_height) / 2.0;
            if overshoot > max_overshoot {
                max_overshoot = overshoot;
            }
        }
    }
    MathRowExtra {
        top: max_overshoot,
        bottom: max_overshoot,
    }
}

/// Combine math and image row extras for one shaped line. Both
/// kinds of overlay can contribute to the same row (a mixed-content
/// paragraph might have inline math next to inline icons); take the
/// max on each side so the row reserves enough vertical space for
/// the taller of the two.
fn combined_row_extra(math: MathRowExtra, image: MathRowExtra) -> MathRowExtra {
    MathRowExtra {
        top: math.top.max(image.top),
        bottom: math.bottom.max(image.bottom),
    }
}

/// Append fallback inline runs for a math overlay whose typeset
/// failed (or whose source slice was unexpectedly out of bounds).
/// The construct's source bytes shape as raw text — this just
/// styles them: dim for the `$` / `$$` delimiter runs, code (mono +
/// faint background) for the content. Mirrors the cursor-inside
/// branch in `render::walk_inline` so the visual is identical
/// whether the user is actively editing the construct or simply
/// looking at a malformed expression.
fn push_failed_overlay_fallback(block: &mut RenderBlock, overlay: &MathOverlay) {
    let opener = overlay.source_range.start..overlay.content_range.start;
    let closer = overlay.content_range.end..overlay.source_range.end;
    for delim in [opener, closer] {
        if !delim.is_empty() {
            block.inlines.push(InlineRun {
                source_range: delim,
                style: InlineStyle::dimmed(),
            });
        }
    }
    if overlay.content_range.start < overlay.content_range.end {
        block.inlines.push(InlineRun {
            source_range: overlay.content_range.clone(),
            style: InlineStyle::code(),
        });
    }
}

/// Shape one [`MATH_PAD_CHAR`] at the block's body font and report
/// its advance — used to size math substitutions in
/// [`augment_block_with_math`].
fn measure_pad_advance(
    em_px: Pixels,
    style: &MarkdownStyle,
    kind: &BlockKind,
    window: &mut Window,
) -> Pixels {
    let font = base_font_for_block(kind, style);
    let s: String = MATH_PAD_CHAR.to_string();
    let runs = [TextRun {
        len: s.len(),
        font,
        color: gpui::black(),
        background_color: None,
        underline: None,
        strikethrough: None,
    }];
    let line = window
        .text_system()
        .shape_text(SharedString::from(s), em_px, &runs, None, None)
        .ok()
        .and_then(|mut v| v.drain(..).next());
    line.map(|l| l.width()).unwrap_or(em_px * 0.2)
}

/// The min-content width of a cell: the widest whitespace-delimited
/// token of its display text, shaped with the token's real styles.
/// This is the floor a column can shrink to before words would have
/// to break mid-word (HTML's min-content).
fn min_content_width(
    range: &Range<usize>,
    block: &RenderBlock,
    source: &str,
    style: &MarkdownStyle,
    font_size: Pixels,
    window: &mut Window,
) -> Pixels {
    let (display, map) = build_display_line(source, range, block);
    if display.is_empty() {
        return px(0.0);
    }
    let mut widest = px(0.0);
    let bytes = display.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let token = &display[start..i];
        let token_map = &map[start..=i];
        // `build_runs_for_line` expects `map.len() == text.len() + 1`.
        debug_assert_eq!(token_map.len(), token.len() + 1);
        let runs = build_runs_for_line(token, token_map, block, style);
        let w = window
            .text_system()
            .shape_text(
                SharedString::from(token.to_string()),
                font_size,
                &runs,
                None,
                None,
            )
            .ok()
            .and_then(|mut v| v.drain(..).next())
            .map(|l| l.width())
            .unwrap_or(px(0.0));
        if w > widest {
            widest = w;
        }
    }
    widest
}

/// One shaped piece of a laid-out table: a cell (independently
/// wrapped at its column's width) or a chrome fragment (the real
/// `| ` / ` | ` / ` |` bytes, single-row, positioned in the gutter
/// between column boxes). Origins are relative to the table's
/// content origin; prepaint re-bases them into window coordinates.
struct TablePiece {
    line: Arc<WrappedLine>,
    rel_origin: Point<Pixels>,
    source_range: Range<usize>,
    display_to_source: Vec<usize>,
    wrapped_height: Pixels,
}

/// The box-model layout of one table.
struct TableLayoutOut {
    pieces: Vec<TablePiece>,
    size: Size<Pixels>,
    /// Relative y of each inter-row boundary (display mode only —
    /// the hairline rules; `[0]` is the header rule).
    rule_ys: Vec<Pixels>,
}

/// Lay a `BlockKind::Table` out as a grid of **cell boxes**: every
/// cell is its own shaped line, wrapped independently at its
/// column's width, positioned side by side — the HTML-auto-inspired
/// model Mike asked for, unified across display and edit mode (and
/// across fitting and overflowing tables, so there is exactly one
/// table layout path).
///
/// **Column sizing:** per column, min-content (the widest unbreakable
/// atom — measured by shaping at a 1px wrap width) and max-content
/// (unwrapped) widths. If Σmax + chrome fits the available measure,
/// every column sits at max-content and nothing wraps (the original
/// behavior, now with exact alignment and no pad glyphs). Otherwise
/// columns shrink proportionally to their `(max − min)` excess,
/// flooring at min-content; if even Σmin overflows, columns floor at
/// min and the table overflows into the shared horizontal-scroll
/// treatment (degenerate narrow window).
///
/// **Rows:** one physical GFM line each, always — wrapping is purely
/// visual. Row height = the tallest wrapped cell. In edit mode the
/// chrome bytes (pipes + padding, including any just-typed trailing
/// whitespace) shape as real single-row fragments in the gutters
/// between boxes, so every byte keeps a true caret position; in
/// display mode the chrome is hidden and fragments shape empty
/// (kept for source coverage). The delimiter row renders only in
/// edit mode, its dash cells stretched to the column width (raw on
/// the caret's row for colon editing). Alignment colons shift a
/// non-wrapped cell inside its box; wrapped cells left-align.
fn layout_table(
    block: &RenderBlock,
    source: &str,
    font_size: Pixels,
    style: &MarkdownStyle,
    avail_w: Pixels,
    window: &mut Window,
) -> Option<TableLayoutOut> {
    let BlockKind::Table {
        geometry,
        edit_mode,
        cursor_row,
    } = &block.kind
    else {
        return None;
    };
    let edit = *edit_mode;
    let cursor_row = *cursor_row;
    let line_height = font_size * style.line_height.0;

    use crate::syntax::TableAlignment;
    use crate::table::RowKind;

    let rows: Vec<(usize, &crate::table::RowGeometry)> = geometry
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| edit || r.kind != RowKind::Delimiter)
        .collect();
    if rows.is_empty() {
        return None;
    }

    // Chrome glyph advances.
    let text_w = |s: &str, window: &mut Window| -> Pixels {
        if s.is_empty() {
            return px(0.0);
        }
        let runs = [TextRun {
            len: s.len(),
            font: base_font_for_block(&block.kind, style),
            color: gpui::black(),
            background_color: None,
            underline: None,
            strikethrough: None,
        }];
        window
            .text_system()
            .shape_text(
                SharedString::from(s.to_string()),
                font_size,
                &runs,
                None,
                None,
            )
            .ok()
            .and_then(|mut v| v.drain(..).next())
            .map(|l| l.width())
            .unwrap_or(px(0.0))
    };
    let w_pipe = text_w("|", window);
    let w_sp = text_w(" ", window);
    let w_dash = text_w("-", window);
    let w_colon = text_w(":", window);
    let gutter = font_size * 1.25;

    // Shape helper over a source slice of the block (hides +
    // substitutions + inline styles apply).
    let shape_slice = |range: &Range<usize>,
                       wrap: Option<Pixels>,
                       window: &mut Window|
     -> Option<(Arc<WrappedLine>, Vec<usize>)> {
        let (display, map) = build_display_line(source, range, block);
        let runs = build_runs_for_line(&display, &map, block, style);
        let line = window
            .text_system()
            .shape_text(SharedString::from(display), font_size, &runs, wrap, None)
            .ok()
            .and_then(|mut v| v.drain(..).next())
            .map(Arc::new)
            .or_else(|| empty_shaped_line(font_size, window))?;
        Some((line, map))
    };

    // Column min/max content widths (delimiter rows excluded — their
    // dash cells stretch to fit). Min-content is the widest
    // *unbreakable atom* — the longest whitespace-delimited token,
    // measured with its real styles (gpui's wrapper falls back to
    // mid-word breaking below a word's width, so shaping at a tiny
    // wrap width would report ~one glyph and let columns squeeze
    // into vertical letter-stacks).
    let n_cols = geometry.column_count().max(1);
    let mut min_w: Vec<Pixels> = vec![px(0.0); n_cols];
    let mut max_w: Vec<Pixels> = vec![px(0.0); n_cols];
    for (_, row) in &rows {
        if row.kind == RowKind::Delimiter {
            continue;
        }
        for (k, cell) in row.cells.iter().enumerate() {
            if k >= n_cols {
                break;
            }
            let (unwrapped, _) = shape_slice(cell, None, window)?;
            if unwrapped.width() > max_w[k] {
                max_w[k] = unwrapped.width();
            }
            let atom_w = min_content_width(cell, block, source, style, font_size, window);
            if atom_w > min_w[k] {
                min_w[k] = atom_w;
            }
        }
    }

    // Available width for the column boxes.
    let chrome_total = if edit {
        // `| ` + per-inner-boundary ` | ` + trailing ` |`.
        (w_pipe + w_sp) + (w_sp + w_pipe + w_sp) * ((n_cols - 1) as f32) + (w_sp + w_pipe)
    } else {
        gutter * (n_cols.saturating_sub(1) as f32)
    };
    let w_avail = (avail_w - chrome_total).max(px(0.0));
    let sum_max: Pixels = max_w.iter().fold(px(0.0), |a, w| a + *w);
    let sum_min: Pixels = min_w.iter().fold(px(0.0), |a, w| a + *w);
    let col_w: Vec<Pixels> = if sum_max <= w_avail || !f32::from(avail_w).is_finite() {
        max_w.clone()
    } else if sum_min <= w_avail {
        // Shrink proportionally to each column's (max − min) excess.
        let excess: Pixels = sum_max - w_avail;
        let flex: Pixels = sum_max - sum_min;
        max_w
            .iter()
            .zip(min_w.iter())
            .map(|(mx, mn)| {
                let share = if flex > px(0.0) {
                    (*mx - *mn) * (f32::from(excess) / f32::from(flex))
                } else {
                    px(0.0)
                };
                (*mx - share).max(*mn)
            })
            .collect()
    } else {
        min_w.clone()
    };
    let col_w: Vec<Pixels> = col_w.into_iter().map(|w| w.max(px(1.0))).collect();

    // Column box x positions.
    let mut col_x: Vec<Pixels> = Vec::with_capacity(n_cols);
    let mut acc = if edit { w_pipe + w_sp } else { px(0.0) };
    for w in &col_w {
        col_x.push(acc);
        acc += *w + if edit { w_sp + w_pipe + w_sp } else { gutter };
    }
    // Total content width: last box end + trailing chrome.
    let total_w = if n_cols > 0 {
        let last_end = col_x[n_cols - 1] + col_w[n_cols - 1];
        if edit {
            last_end + w_sp + w_pipe
        } else {
            last_end
        }
    } else {
        px(0.0)
    };

    let mut pieces: Vec<TablePiece> = Vec::new();
    let mut rule_ys: Vec<Pixels> = Vec::new();
    let mut y = px(0.0);
    let last_row_idx = rows.len() - 1;
    for (i, (row_geo_idx, row)) in rows.iter().enumerate() {
        let is_delimiter = row.kind == RowKind::Delimiter;
        let stretch = is_delimiter && cursor_row != Some(*row_geo_idx);
        let mut row_h = line_height;

        // Cell boxes.
        for (k, cell) in row.cells.iter().enumerate() {
            let bw = col_w.get(k).copied().unwrap_or(px(64.0));
            let bx = col_x.get(k).copied().unwrap_or(px(0.0));
            let (line, map, ox) = if stretch && cell.start < cell.end {
                // Stretched delimiter cell: dash run sized to the box.
                let cell_text = &source[cell.clone()];
                let leading = cell_text.starts_with(':');
                let trailing = cell_text.ends_with(':') && cell_text.len() > 1;
                let colons = ((leading as usize + trailing as usize) as f32) * w_colon;
                let n_dashes = if w_dash > px(0.0) {
                    (((bw - colons) / w_dash).floor() as i64).max(1) as usize
                } else {
                    3
                };
                let mut display = String::new();
                if leading {
                    display.push(':');
                }
                display.extend(std::iter::repeat_n('-', n_dashes));
                if trailing {
                    display.push(':');
                }
                let map = vec![cell.start; display.len() + 1];
                let runs = [TextRun {
                    len: display.len(),
                    font: base_font_for_block(&block.kind, style),
                    color: style.delimiter_color,
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                }];
                let line = window
                    .text_system()
                    .shape_text(SharedString::from(display), font_size, &runs, None, None)
                    .ok()
                    .and_then(|mut v| v.drain(..).next())
                    .map(Arc::new)
                    .or_else(|| empty_shaped_line(font_size, window))?;
                (line, map, bx)
            } else {
                // Measure unwrapped; wrap only when it doesn't fit.
                let (unwrapped, map) = shape_slice(cell, None, window)?;
                if unwrapped.width() <= bw + px(0.5) {
                    // Alignment shift inside the box (single-row
                    // cells only; wrapped cells left-align).
                    let free = (bw - unwrapped.width()).max(px(0.0));
                    let fill = if is_delimiter {
                        px(0.0)
                    } else {
                        match geometry.alignment(k) {
                            TableAlignment::Right => free,
                            TableAlignment::Center => free * 0.5,
                            TableAlignment::None | TableAlignment::Left => px(0.0),
                        }
                    };
                    (unwrapped, map, bx + fill)
                } else {
                    let (wrapped, map) = shape_slice(cell, Some(bw), window)?;
                    (wrapped, map, bx)
                }
            };
            let h = line_height * (line.wrap_boundaries().len() as f32 + 1.0);
            if h > row_h {
                row_h = h;
            }
            // The cell claims one byte past its content end, and cell
            // pieces precede the row's chrome fragments in the vec:
            // the caret at a cell's content end (the typing position,
            // shared with the following fragment's start) therefore
            // resolves to the CELL and paints at its text end — not at
            // the column box's far edge where the fragment starts.
            // Offsets strictly inside the padding still resolve to the
            // fragment (true x in the gutter).
            let claim_end = (cell.end + 1).min(source.len());
            pieces.push(TablePiece {
                line,
                rel_origin: point(ox, y),
                source_range: cell.start..claim_end,
                display_to_source: map,
                wrapped_height: h,
            });
        }

        // Chrome fragments: [line start .. cell0 start], each
        // inter-cell gap, and the trailing chunk (which swallows the
        // row's `\n` so end-of-line offsets stay covered).
        let mut frag_ranges: Vec<(Range<usize>, Pixels)> = Vec::new();
        if let Some(first) = row.cells.first() {
            frag_ranges.push((row.line.start..first.start, px(0.0)));
            for k in 1..row.cells.len() {
                let x = col_x.get(k - 1).copied().unwrap_or(px(0.0))
                    + col_w.get(k - 1).copied().unwrap_or(px(0.0));
                frag_ranges.push((row.cells[k - 1].end..row.cells[k].start, x));
            }
            let k_last = row.cells.len() - 1;
            let trail_x = col_x.get(k_last).copied().unwrap_or(px(0.0))
                + col_w.get(k_last).copied().unwrap_or(px(0.0));
            let mut trail_end = row.line.end;
            if source.as_bytes().get(trail_end) == Some(&b'\n') {
                trail_end += 1;
            }
            frag_ranges.push((row.cells[k_last].end..trail_end, trail_x));
        }
        for (range, x) in frag_ranges {
            if range.start >= range.end {
                continue;
            }
            let (line, map) = shape_slice(&range, None, window)?;
            pieces.push(TablePiece {
                line,
                rel_origin: point(x, y),
                source_range: range,
                display_to_source: map,
                wrapped_height: line_height,
            });
        }

        // The row's pieces all share the row top; the row advances by
        // its tallest piece. (Fragments keep `line_height` — they
        // never wrap.)
        y += row_h;
        if !edit && i < last_row_idx {
            rule_ys.push(y);
        }
    }

    Some(TableLayoutOut {
        pieces,
        size: size(total_w, y),
        rule_ys,
    })
}

// ---------------------------------------------------------------------------
// Embed blocks — layout + paint of a `BlockKind::Embed`'s mapped markdown
// (see `crate::embed` for the plugin contract). The mapped content is
// rendered read-only: parsed and run through `render_readonly` (every
// delimiter hidden, exactly like a disabled editor), then each sub-block is
// shaped with the same machinery the editor uses for its own blocks and
// stacked vertically inside a quiet quote-like container (left border bar +
// blockquote indent). The sub-lines are display-only — no cursor, no
// hit-testing inside; the *outer* block's atomicity rules own interaction.
//
// Chrome that isn't shaped text has to be re-emitted here explicitly, since
// only the shaping path is shared: list markers (`emit_embed_marker_overlays`
// — bullets, numbers, task checkboxes), code-block panels, thematic-break and
// table rules, nested blockquote bars, and typeset math.
//
// Remaining fidelity notes: an embedded table reuses `layout_table` but takes
// no horizontal scroll (a wide one clips at the embed mask, as does a long
// code line); a code block gets one flat panel rather than the editor's
// fence-frame-plus-inset-strip, because the embed collapses the fence rows
// that frame would need; and **images render as their raw markdown source**
// (`strip_embed_image_overlays` un-hides their bytes so nothing silently
// vanishes) — resolving one calls `Window::use_asset`, which needs the
// rendering entity on the window stack, and `layout_embed` also runs from the
// measure closure, where it isn't.
// ---------------------------------------------------------------------------

/// One shaped line of an embed's rendered content. Origins are relative to
/// the embed's content origin; prepaint re-bases them into window
/// coordinates.
struct EmbedPiece {
    line: Arc<WrappedLine>,
    rel_origin: Point<Pixels>,
    row_height: Pixels,
}

/// One typeset display-math sub-block of an embed.
struct EmbedMathPiece {
    rel_origin: Point<Pixels>,
    layout: crate::math::MathLayout,
    em_px: Pixels,
}

/// The stacked layout of an embed's mapped markdown.
struct EmbedLayoutOut {
    pieces: Vec<EmbedPiece>,
    math: Vec<EmbedMathPiece>,
    /// Hairline rules (embedded tables' row boundaries, thematic breaks):
    /// relative bounds + full-strength flag (header rules paint at full
    /// `table_rule_color`; the rest lighter, mirroring the table paint).
    rules: Vec<(Bounds<Pixels>, bool)>,
    /// Blockquote border bars for nested blockquotes *inside* the embedded
    /// content, relative bounds.
    bars: Vec<Bounds<Pixels>>,
    /// Rounded background panels for embedded fenced code blocks. One flat
    /// fill rather than the main editor's two-layer fence-frame-plus-inset-
    /// strip, because an embed collapses the fence rows that frame would
    /// need — the panel *is* the content area.
    code_panels: Vec<Bounds<Pixels>>,
    size: Size<Pixels>,
}

/// Un-hide the byte ranges of image overlays and drop the records, so the raw
/// markdown shapes honestly instead of silently vanishing under a hide with
/// nothing painted in its place.
///
/// Images are the one overlay an embed can't resolve: `crate::image::load`
/// calls `Window::use_asset`, which needs the rendering entity on the window's
/// stack — and `layout_embed` also runs from the measure closure, after
/// `request_layout` has returned. Inline math has no such constraint and does
/// run its augment pass inside embeds.
fn strip_embed_image_overlays(block: &mut RenderBlock) {
    let cuts: Vec<Range<usize>> = block
        .image_overlays
        .drain(..)
        .map(|o| o.source_range)
        .collect();
    for cut in &cuts {
        subtract_hidden_range(&mut block.hidden_ranges, cut);
    }
}

/// Remove `cut` from every range in `hidden`, splitting ranges that
/// straddle it.
fn subtract_hidden_range(hidden: &mut Vec<Range<usize>>, cut: &Range<usize>) {
    let mut out = Vec::with_capacity(hidden.len());
    for h in hidden.drain(..) {
        if h.end <= cut.start || h.start >= cut.end {
            out.push(h);
            continue;
        }
        if h.start < cut.start {
            out.push(h.start..cut.start);
        }
        if h.end > cut.end {
            out.push(cut.end..h.end);
        }
    }
    *hidden = out;
}

/// Position every inline-math construct that lives on `row` as an embed math
/// piece. The embed twin of `paint_inline_math_overlays`: the substitution's
/// display offset gives x; the y puts the math's baseline on the row's text
/// baseline, mirroring gpui's vertical centering of a shaped line inside
/// `row_height`.
///
/// Consumes the specs it places (`MathLayout` isn't `Clone`, and a construct
/// belongs to exactly one row anyway) — the caller's vector keeps only the
/// specs still awaiting a row.
#[allow(clippy::too_many_arguments)]
fn place_embed_inline_math(
    specs: &mut Vec<InlineMathPaintSpec>,
    row: &ShapedLine,
    row_origin: Point<Pixels>,
    row_height: Pixels,
    body_ascent: Pixels,
    body_descent: Pixels,
    out: &mut Vec<EmbedMathPiece>,
) {
    if specs.is_empty() {
        return;
    }
    let half_leading = ((row_height - body_ascent - body_descent) / 2.0).max(px(0.0));
    let mut unplaced = Vec::with_capacity(specs.len());
    for spec in specs.drain(..) {
        let on_this_row = spec.source_range.start >= row.source_range.start
            && spec.source_range.start < row.source_range.end;
        let local = on_this_row
            .then(|| {
                row.display_to_source
                    .iter()
                    .position(|&src| src == spec.source_range.start)
                    .and_then(|off| row.line.position_for_index(off, row_height))
            })
            .flatten();
        let Some(local) = local else {
            unplaced.push(spec);
            continue;
        };
        let text_baseline = row_origin.y + local.y + half_leading + body_ascent;
        out.push(EmbedMathPiece {
            rel_origin: point(
                row_origin.x + local.x,
                text_baseline - spec.layout.baseline(spec.em_px),
            ),
            layout: spec.layout,
            em_px: spec.em_px,
        });
    }
    *specs = unplaced;
}

/// Re-emit a block's [`MarkerOverlay`](crate::render_spec::MarkerOverlay)
/// chrome as embed pieces.
///
/// A list item's marker (`• `, `1. `, `☑ `) is **not** shaped text: the
/// render layer hides the marker bytes so content shapes at column 0, and the
/// element paints the glyph into the item's indent strip. `layout_embed`
/// shapes lines only, so before this every embedded list rendered as bare
/// indented paragraphs — no bullets, no numbers, no checkboxes.
///
/// Positioning mirrors `paint_list_marker_overlay` exactly (right-aligned
/// against the level's content edge, so short and long markers in one list
/// share a content edge); the row comes from the shaped row whose source
/// range contains the marker's first byte, falling back to the block's top.
///
/// Blockquote markers are deliberately not handled: `render::attach_marker`
/// records them only when the cursor is *inside*, and embed content renders
/// through `render_readonly` (cursor nowhere), so an embed never has one.
/// The bar itself is drawn by `layout_embed`'s container pass.
fn emit_embed_marker_overlays(
    block: &RenderBlock,
    shaped_rows: &[(Range<usize>, Pixels)],
    block_top: Pixels,
    line_height: Pixels,
    style: &MarkdownStyle,
    window: &mut Window,
    pieces: &mut Vec<EmbedPiece>,
) {
    for marker in &block.marker_overlays {
        let Some(Container::ListItem {
            kind,
            cursor_inside,
            ..
        }) = block.containers.get(marker.level)
        else {
            continue;
        };
        let text = list_marker_display_text(*kind, *cursor_inside);
        let Some(glyph) = shape_marker_text(&text, style, window) else {
            continue;
        };
        let strip_right = container_x_at_level(&block.containers, marker.level + 1, style, window);
        let row_y = shaped_rows
            .iter()
            .find(|(r, _)| {
                r.start <= marker.source_range.start && marker.source_range.start < r.end
            })
            .map(|(_, y)| *y)
            .unwrap_or(block_top);
        pieces.push(EmbedPiece {
            rel_origin: point((strip_right - glyph.width()).max(px(0.0)), row_y),
            line: glyph,
            row_height: line_height,
        });
    }
}

/// Lay out an embed's mapped markdown at `avail_w`: parse + readonly-render
/// the content, then stack each sub-block using the editor's own shaping and
/// spacing rules (per-block font size, symmetric half-gaps, container
/// indents). Pure function of its inputs — the measure closure and prepaint
/// both call it so the two phases can't drift.
fn layout_embed(
    content: &str,
    style: &MarkdownStyle,
    avail_w: Pixels,
    window: &mut Window,
) -> EmbedLayoutOut {
    let state = crate::state::EditorState::with_markdown(content.to_string());
    let tree = crate::parser::parse(content);
    let spec = crate::render::render_readonly(&state, &tree);

    let mut out = EmbedLayoutOut {
        pieces: Vec::new(),
        math: Vec::new(),
        rules: Vec::new(),
        bars: Vec::new(),
        code_panels: Vec::new(),
        size: Size::default(),
    };
    let mut y = px(0.0);
    let mut max_w = px(0.0);
    for block in &spec.blocks {
        let font_size = font_size_for_block(&block.kind, style);
        let line_height = font_size * style.line_height.0;
        let above = spacing_above_for_block(&block.kind, &block.containers, style);
        let below = spacing_below_for_block(&block.kind, &block.containers, style);
        let indent = containers_left_indent(&block.containers, style, window);
        let inner_w = (avail_w - indent).max(px(1.0));
        let bar_top = y;
        y += above;
        let block_top = y;

        // Source range + y origin of each shaped row this block emitted —
        // what `emit_embed_marker_overlays` needs to put a list marker on the
        // row its bytes belong to (the main editor reads the same thing off
        // `LaidOutLine`).
        let mut shaped_rows: Vec<(Range<usize>, Pixels)> = Vec::new();

        match &block.kind {
            BlockKind::Table { .. } => {
                if let Some(t) = layout_table(block, content, font_size, style, inner_w, window) {
                    for piece in t.pieces {
                        out.pieces.push(EmbedPiece {
                            line: piece.line,
                            rel_origin: point(
                                indent + piece.rel_origin.x,
                                block_top + piece.rel_origin.y,
                            ),
                            row_height: line_height,
                        });
                    }
                    for (i, ry) in t.rule_ys.iter().enumerate() {
                        out.rules.push((
                            Bounds::new(
                                point(indent, block_top + *ry - px(0.5)),
                                size(t.size.width.max(px(1.0)), px(1.0)),
                            ),
                            i == 0,
                        ));
                    }
                    if indent + t.size.width > max_w {
                        max_w = indent + t.size.width;
                    }
                    y = block_top + t.size.height.max(line_height);
                }
            }
            BlockKind::ThematicBreak => {
                // The source bytes are hidden in readonly render; paint the
                // rule the editor would paint, centered on the row.
                out.rules.push((
                    Bounds::new(
                        point(indent, block_top + line_height * 0.5 - px(0.5)),
                        size(inner_w, px(1.0)),
                    ),
                    false,
                ));
                y = block_top + line_height;
            }
            _ => {
                // Display math: typeset like the editor's display mode; a
                // typeset failure (malformed LaTeX) falls through to the raw
                // text path with the block hide cleared so the source shows.
                let mut handled_math = false;
                if matches!(block.kind, BlockKind::DisplayMath { .. })
                    && let Some(layout) = typeset_display_math_block(block, content)
                {
                    let msize = layout.size(font_size);
                    out.math.push(EmbedMathPiece {
                        rel_origin: point(indent, block_top),
                        layout,
                        em_px: font_size,
                    });
                    if indent + msize.width > max_w {
                        max_w = indent + msize.width;
                    }
                    y = block_top + msize.height.max(line_height);
                    handled_math = true;
                }
                if !handled_math {
                    let mut b = block.clone();
                    if matches!(b.kind, BlockKind::DisplayMath { .. }) {
                        b.hidden_ranges.clear();
                    }
                    // Inline math typesets inside an embed exactly as it does
                    // in the editor: the augment pass replaces each construct
                    // with a width-matched padding run and returns a paint
                    // spec. Images can't take this path — see
                    // `strip_embed_image_overlays`.
                    let (mut b, mut math_specs) =
                        augment_block_with_math(&b, content, font_size, style, window);
                    strip_embed_image_overlays(&mut b);
                    let (body_ascent, body_descent) =
                        body_metrics_for_block(&b.kind, style, font_size, window);
                    let is_code = is_code_block(&b.kind);
                    let wrap_w = if is_code { None } else { Some(inner_w) };
                    // A code block insets its content from its background
                    // panel on all four sides, exactly as the main editor
                    // does — without the pad the first and last code rows sit
                    // flush against the panel edge.
                    let pad = if is_code {
                        style.code_block_padding
                    } else {
                        px(0.0)
                    };
                    let text_left = indent + pad;
                    y += pad;
                    let shaped = shape_block_lines(content, &b, style, font_size, wrap_w, window);
                    let mut advanced = false;
                    for sl in shaped {
                        // Fence rows are hidden in readonly render but still
                        // reserve a row in the main editor; inside an embed we
                        // collapse them instead — the quote shows code, not
                        // the fences' blank rows.
                        if sl.is_delimiter {
                            continue;
                        }
                        // A row whose math overshoots the body font's natural
                        // half-leading reserves extra space above/below, the
                        // same shift the editor's own line layout applies —
                        // otherwise a tall construct paints over its
                        // neighbours.
                        let extra = compute_math_row_extra(
                            &sl.source_range,
                            &math_specs,
                            body_ascent,
                            body_descent,
                            line_height,
                        );
                        let wrap_count = (sl.line.wrap_boundaries().len() as f32) + 1.0;
                        let h = (line_height + extra.total()) * wrap_count;
                        let row_y = y + extra.top;
                        if text_left + sl.line.width() > max_w {
                            max_w = text_left + sl.line.width();
                        }
                        place_embed_inline_math(
                            &mut math_specs,
                            &sl,
                            point(text_left, row_y),
                            line_height,
                            body_ascent,
                            body_descent,
                            &mut out.math,
                        );
                        shaped_rows.push((sl.source_range.clone(), row_y));
                        out.pieces.push(EmbedPiece {
                            line: sl.line,
                            rel_origin: point(text_left, row_y),
                            row_height: line_height,
                        });
                        y += h;
                        advanced = true;
                    }
                    if !advanced {
                        // Keep the layout total even for a blank block.
                        y = block_top + line_height + pad;
                    }
                    y += pad;
                    if is_code {
                        out.code_panels.push(Bounds::new(
                            point(indent, block_top),
                            size(inner_w, (y - block_top).max(px(0.0))),
                        ));
                        if indent + inner_w > max_w {
                            max_w = indent + inner_w;
                        }
                    }
                }
            }
        }

        emit_embed_marker_overlays(
            block,
            &shaped_rows,
            block_top,
            line_height,
            style,
            window,
            &mut out.pieces,
        );

        y += below;

        // Nested blockquote bars span the block's layout box (half-gaps
        // included) so consecutive quoted leaves read as one bar.
        for (level, container) in block.containers.iter().enumerate() {
            if matches!(container, Container::BlockQuote { .. }) {
                let left = container_x_at_level(&block.containers, level, style, window)
                    + style.blockquote_border_inset;
                out.bars.push(Bounds::new(
                    point(left, bar_top),
                    size(style.blockquote_border_width, y - bar_top),
                ));
            }
        }
    }

    out.size = size(max_w, y);
    out
}

/// Paint state for a `BlockKind::Embed` block: the re-based (window-coord)
/// content pieces plus the quote chrome.
struct EmbedPaint {
    pieces: Vec<EmbedPiece>,
    math: Vec<EmbedMathPiece>,
    rules: Vec<(Bounds<Pixels>, bool)>,
    bars: Vec<Bounds<Pixels>>,
    code_panels: Vec<Bounds<Pixels>>,
    /// The embed's own quote bar (the container chrome).
    bar: Bounds<Pixels>,
    /// Full content bounds — the clip mask and the selection-wash rect.
    bounds: Bounds<Pixels>,
    /// True when the outer selection covers the whole marker — the visual
    /// "selected as one unit" cue.
    selected: bool,
}

/// Paint state for a `BlockKind::Table` block: horizontal-scroll
/// geometry (tables never soft-wrap; wide ones scroll like code
/// blocks) plus the hairline rules of display mode.
struct TablePaint {
    /// Content clip when the table overflows the visible width.
    clip: Bounds<Pixels>,
    /// Widest shaped table line.
    content_width: Pixels,
    visible_width: Pixels,
    scroll_x: Pixels,
    /// Hairline rule quads (positioned, already scroll-translated).
    /// `rules[0]` is the header rule when present.
    rules: Vec<Bounds<Pixels>>,
}

#[derive(Debug, Clone, Copy)]
struct CodeBlockPaint {
    /// Outer rounded fill (full block, fence rows + content strip).
    bg_bounds: Bounds<Pixels>,
    /// Inner content strip — full-width band between the fence rows.
    /// A second (slightly different) bg paints edge-to-edge here so
    /// the content area visually inverts the fence frame.
    content_strip: Bounds<Pixels>,
    /// Sub-rectangle of `content_strip` inset by `inner_pad`
    /// horizontally. Used as the `content_mask` for content lines so
    /// long lines clip at the inner padding edge instead of leaking
    /// past the rounded outer fill.
    content_clip: Bounds<Pixels>,
    content_width: Pixels,
    visible_width: Pixels,
    scroll_x: Pixels,
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
        let spacing_above =
            spacing_above_for_block(&self.block.kind, &self.block.containers, &self.style);
        let spacing_below =
            spacing_below_for_block(&self.block.kind, &self.block.containers, &self.style);
        let extra_above = container_boundary_extra(
            &self.block.containers,
            self.prev_containers.as_deref(),
            &self.style,
        );
        let extra_below = container_boundary_extra(
            &self.block.containers,
            self.next_containers.as_deref(),
            &self.style,
        );
        let inner_pad = block_inner_padding(&self.block.kind, &self.style);
        let is_code = is_code_block(&self.block.kind);
        let is_table = is_table_block(&self.block.kind);
        let container_indent = containers_left_indent(&self.block.containers, &self.style, window);

        let source = self.editor.read(cx).state.markdown.clone();
        style.size.width = relative(1.0).into();

        // Pre-resolve every image referenced by this block *before*
        // the measure closure runs. `crate::image::load` ultimately
        // calls `Window::use_asset`, which on first-cache-miss reads
        // `Window::current_view()` to register a notify callback —
        // and `current_view()` panics if the currently-rendering
        // entity isn't on the window's stack. Measure callbacks run
        // *after* `request_layout` returns (taffy invokes them
        // during layout), at which point the entity is no longer on
        // the stack. By doing the load here (inside `request_layout`
        // proper, where the entity is on the stack), we both
        // sidestep the panic and ensure the editor view is wired up
        // to re-render once the asset loads.
        let inline_image_loaded: Vec<crate::image::LoadedImage> = self
            .block
            .image_overlays
            .iter()
            .map(|o| crate::image::load(&o.dest_url, window, cx))
            .collect();
        // Pre-load the block image in *both* display and edit mode.
        // Edit mode also needs the image's natural size to honor the
        // edit-mode min-height rule (block reserves max of the natural
        // edit-shape height and the natural display-paint height) so
        // toggling edit mode doesn't shift surrounding content.
        let block_image_loaded: Option<crate::image::LoadedImage> =
            if let BlockKind::Image { dest_url, .. } = &self.block.kind {
                Some(crate::image::load(dest_url, window, cx))
            } else {
                None
            };

        // Embed content is resolved outside the measure closure (it reads
        // the editor entity). `None` for non-embed blocks; an Embed block
        // with a missing map entry cannot occur (promotion requires the
        // mapping) but degrades to a one-line blank block if it ever did.
        let embed_content: Option<String> = if let BlockKind::Embed { ordinal } = &self.block.kind {
            self.editor
                .read(cx)
                .state
                .embeds
                .get(*ordinal)
                .map(str::to_string)
        } else {
            None
        };

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
                // Display-math in display mode bypasses text shaping
                // entirely — the height is whatever the typeset math
                // requires. A typesetting failure (malformed LaTeX)
                // falls through to a one-line height so the user can
                // still see the cursor and edit the source.
                if matches!(
                    block_clone.kind,
                    BlockKind::DisplayMath {
                        edit_mode: false,
                        ..
                    }
                ) && let Some(math_layout) = typeset_display_math_block(&block_clone, &source)
                {
                    let size = math_layout.size(font_size);
                    let h = extra_above
                        + spacing_above
                        + size.height.max(line_height)
                        + spacing_below
                        + extra_below;
                    return Size {
                        width: avail_w,
                        height: h,
                    };
                }
                // Image block (display mode): height comes from the
                // loaded image's natural size, scaled to fit
                // `avail_w`. Mirrors the display-math fast path. A
                // failed load falls through to the regular shape
                // path so the dim-delim + alt-text fallback the
                // render layer pre-staged shows up underneath. The
                // image was pre-resolved outside this closure —
                // see the load comment above `request_measured_layout`.
                if let BlockKind::Image {
                    edit_mode: false, ..
                } = &block_clone.kind
                {
                    let inner_w = (avail_w - container_indent).max(px(1.0));
                    let loaded = block_image_loaded
                        .as_ref()
                        .cloned()
                        .unwrap_or(crate::image::LoadedImage::Loading);
                    let size_opt = match loaded {
                        crate::image::LoadedImage::Loaded(img) => {
                            Some(crate::image::block_size(img.as_ref(), inner_w))
                        }
                        crate::image::LoadedImage::Loading => {
                            Some(crate::image::block_placeholder_size(font_size, inner_w))
                        }
                        crate::image::LoadedImage::Failed => None,
                    };
                    if let Some(size) = size_opt {
                        let h = extra_above
                            + spacing_above
                            + size.height.max(line_height)
                            + spacing_below
                            + extra_below;
                        return Size {
                            width: avail_w,
                            height: h,
                        };
                    }
                    // Failed: fall through to the regular shape path so
                    // the raw `![alt](url)` source shapes underneath
                    // the dim-delim + alt-text fallback runs the
                    // render layer staged. The user sees what they
                    // typed and can fix the broken URL.
                }
                // Embed block: height = the mapped markdown's stacked
                // layout inside the quote container (bar inset + indent).
                // Mirrors the display-math fast path — the marker's own
                // shaped line is fully hidden and contributes nothing.
                if matches!(block_clone.kind, BlockKind::Embed { .. }) {
                    let inner_w =
                        (avail_w - container_indent - style_clone.blockquote_indent).max(px(1.0));
                    let content_h = match &embed_content {
                        Some(content) => {
                            layout_embed(content, &style_clone, inner_w, window)
                                .size
                                .height
                        }
                        None => line_height,
                    };
                    let h = extra_above
                        + spacing_above
                        + content_h.max(line_height)
                        + spacing_below
                        + extra_below;
                    return Size {
                        width: avail_w,
                        height: h,
                    };
                }
                // Container chain (blockquotes, future list-items)
                // eats left indent — content shapes within the
                // remaining width so soft-wrap lands at the visible
                // right edge regardless of nesting depth.
                let inner_w = (avail_w - container_indent).max(px(1.0));
                // Code blocks and tables don't soft-wrap — long lines
                // extend off the right edge of the visible region and
                // the user scrolls horizontally to see them. Other
                // blocks wrap at the indented inner width.
                let wrap_w = if is_code || is_table {
                    None
                } else {
                    Some(inner_w)
                };
                // Inline math: typeset each overlay and substitute a
                // width-matched run of NBSPs so wrap math sees the
                // math's pixel width. Paint specs flow through to
                // `shaped_content_height` so the block's reserved
                // height grows for any line whose math is taller
                // than the body font.
                let (augmented_block, paint_specs) =
                    augment_block_with_math(&block_clone, &source, font_size, &style_clone, window);
                // Inline images: same substitution mechanism as math
                // (load → measure → pad-run substitution). The loads
                // themselves happened in `request_layout`'s outer
                // body (see comment above) so this closure only
                // consumes the pre-resolved state.
                let (augmented_block, image_specs) = augment_block_with_images(
                    &augmented_block,
                    &inline_image_loaded,
                    font_size,
                    line_height,
                    &style_clone,
                    window,
                );
                // Tables: the box-model layout computes its own
                // height (rows of independently wrapped cell boxes);
                // it must run against the same inner width prepaint
                // will use so the two phases agree.
                if is_table
                    && let Some(layout) = layout_table(
                        &augmented_block,
                        &source,
                        font_size,
                        &style_clone,
                        inner_w,
                        window,
                    )
                {
                    let content_h = inner_pad * 2. + layout.size.height.max(line_height);
                    let h = extra_above + spacing_above + content_h + spacing_below + extra_below;
                    return Size {
                        width: avail_w,
                        height: h,
                    };
                }
                let (body_ascent, body_descent) =
                    body_metrics_for_block(&block_clone.kind, &style_clone, font_size, window);
                let lines = shape_block_lines(
                    &source,
                    &augmented_block,
                    &style_clone,
                    font_size,
                    wrap_w,
                    window,
                );
                // Block height is `extra_above` + `spacing_above` +
                // 2× `inner_pad` + (sum of shaped line heights with
                // code-block breathing) + `spacing_below` +
                // `extra_below`. The same content arithmetic runs in
                // `prepaint` to position lines — extracted into
                // `shaped_content_height` so the two phases can't
                // drift on wrap math, breathing pads, the empty-
                // block fallback, or per-line math row padding.
                // `spacing_below` is part of the layout box so
                // per-block decorations spanning `bounds` minus the
                // extras (e.g. the blockquote border bar) extend
                // symmetric around the content rows; the extras
                // themselves sit *outside* the decoration so they
                // read as inter-block breathing room rather than
                // part of the construct.
                let mut content_h = inner_pad * 2.
                    + shaped_content_height(
                        &lines,
                        line_height,
                        &paint_specs,
                        &image_specs,
                        body_ascent,
                        body_descent,
                        is_code,
                        &style_clone,
                    );
                // Edit-mode min-height: a `BlockKind::DisplayMath` /
                // `BlockKind::Image` in edit mode reserves at least
                // the height the *display*-mode paint would take, so
                // the user's vertical context doesn't shift when the
                // cursor enters or leaves the construct. The display
                // height is computed here from the same typeset /
                // image-load that display mode uses; on a typeset
                // failure (malformed LaTeX) or a Failed image load we
                // fall through with no min — the construct's edit-mode
                // shape *is* what display mode would show as fallback.
                let display_min_height = match &block_clone.kind {
                    BlockKind::DisplayMath {
                        edit_mode: true, ..
                    } => typeset_display_math_block(&block_clone, &source)
                        .map(|layout| layout.size(font_size).height),
                    BlockKind::Image {
                        edit_mode: true, ..
                    } => {
                        let inner_w = (avail_w - container_indent).max(px(1.0));
                        match block_image_loaded
                            .as_ref()
                            .cloned()
                            .unwrap_or(crate::image::LoadedImage::Loading)
                        {
                            crate::image::LoadedImage::Loaded(img) => {
                                Some(crate::image::block_size(img.as_ref(), inner_w).height)
                            }
                            crate::image::LoadedImage::Loading => Some(
                                crate::image::block_placeholder_size(font_size, inner_w).height,
                            ),
                            crate::image::LoadedImage::Failed => None,
                        }
                    }
                    _ => None,
                };
                if let Some(min) = display_min_height {
                    let min_total = inner_pad * 2. + min.max(line_height);
                    if content_h < min_total {
                        content_h = min_total;
                    }
                }
                let h = extra_above + spacing_above + content_h + spacing_below + extra_below;
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
        let (selection, source, scroll_offset, caret_downstream, highlight_layers) = {
            let editor = self.editor.read(cx);
            let scroll = editor.code_block_scroll(self.block_index);
            (
                editor.state.selection,
                editor.state.markdown.clone(),
                scroll,
                editor.caret_downstream(),
                editor.highlight_layers().merged_by_layer(),
            )
        };

        let style = self.style.clone();
        let font_size = font_size_for_block(&self.block.kind, &style);
        let line_height = font_size * style.line_height.0;
        let spacing_above =
            spacing_above_for_block(&self.block.kind, &self.block.containers, &style);
        let extra_above = container_boundary_extra(
            &self.block.containers,
            self.prev_containers.as_deref(),
            &style,
        );
        // `extra_below` doesn't affect content positioning during
        // prepaint — it sits past `block_bottom` and is already part
        // of `bounds.size.height` from `request_layout`. The bar paint
        // recomputes it for its own bounds.
        let inner_pad = block_inner_padding(&self.block.kind, &style);
        let is_code = is_code_block(&self.block.kind);
        let is_table = is_table_block(&self.block.kind);
        // Blocks that never soft-wrap and scroll horizontally instead.
        let no_wrap = is_code || is_table;
        let container_indent = containers_left_indent(&self.block.containers, &style, window);

        let block_top = bounds.origin.y + extra_above + spacing_above;
        // `block_left` is the left edge of *this* leaf's visible
        // region — `bounds.origin.x` shifted right by every
        // container's indent. Per-level blockquote borders paint at
        // `bounds.origin.x + i * blockquote_indent` (left of
        // `block_left`); the code-block bg / content all sit at or
        // right of `block_left`. Click-detection bounds still span
        // the full incoming `bounds` so a click on the indent strip
        // still hits this leaf.
        let block_left = bounds.origin.x + container_indent;
        // Code blocks inset their content from the rounded background
        // fill by `inner_pad` on every edge. Non-code blocks have
        // `inner_pad == 0` and behave as before.
        let content_top = block_top + inner_pad;
        let content_left = block_left + inner_pad;
        let block_width = (bounds.size.width - container_indent).max(px(1.0));
        let visible_content_width = (block_width - inner_pad * 2.).max(px(1.0));

        let wrap_w = if no_wrap { None } else { Some(block_width) };
        // Inline math: typeset overlays and substitute width-matched
        // NBSPs into a working clone of the block. Paint specs flow
        // through to the paint phase; the augmented block is what
        // gets shaped.
        let (augmented_block, inline_math_specs) =
            augment_block_with_math(&self.block, &source, font_size, &style, window);
        // Inline images: pre-resolve each overlay via the asset
        // cache (safe in prepaint — the view is on the stack here),
        // then hand the loaded states to the augment pass. Async
        // loads complete via cache-driven view invalidation so a
        // still-`Loading` frame today is a fully-`Loaded` one
        // tomorrow.
        let inline_image_loaded: Vec<crate::image::LoadedImage> = self
            .block
            .image_overlays
            .iter()
            .map(|o| crate::image::load(&o.dest_url, window, cx))
            .collect();
        let (augmented_block, inline_image_specs) = augment_block_with_images(
            &augmented_block,
            &inline_image_loaded,
            font_size,
            line_height,
            &style,
            window,
        );
        let mut augmented_block = augmented_block;
        // Block image in display mode: hide the source bytes when
        // we have a paintable image (Loaded) or a placeholder
        // (Loading) so the raw `![alt](url)` text doesn't shape
        // underneath. On Failed we don't hide — the fallback runs
        // the render layer staged (dim delim + alt) shape on the
        // raw source bytes so the user sees what they typed.
        if let BlockKind::Image {
            edit_mode: false, ..
        } = &self.block.kind
        {
            let loaded = crate::image::load(
                match &self.block.kind {
                    BlockKind::Image { dest_url, .. } => dest_url,
                    _ => unreachable!(),
                },
                window,
                cx,
            );
            if !loaded.is_failed() {
                augmented_block
                    .hidden_ranges
                    .push(self.block.source_range.clone());
            }
        }
        let (body_ascent, body_descent) =
            body_metrics_for_block(&self.block.kind, &style, font_size, window);
        // Tables take the box-model layout (rows of independently
        // wrapped cell boxes — see `layout_table`) instead of the
        // per-source-line shape pass.
        let table_layout = if is_table {
            layout_table(
                &augmented_block,
                &source,
                font_size,
                &style,
                block_width,
                window,
            )
        } else {
            None
        };
        let shaped = if table_layout.is_some() {
            Vec::new()
        } else {
            shape_block_lines(&source, &augmented_block, &style, font_size, wrap_w, window)
        };

        let mut lines: Vec<LaidOutLine> = Vec::new();
        let mut content_cursor_y = content_top;
        // Track the widest *content* line — fence lines are pinned and
        // never participate in overflow / scrollbar arithmetic.
        let mut max_content_line_width = px(0.0);
        // Vertical extent of the inner content strip. Tracked as the
        // y just *after* the opener fence (where the strip's top
        // edge sits) through the y just *before* the closer fence
        // (where its bottom edge sits). Includes the per-side
        // `code_block_content_padding_y` that breathes between the
        // fence rows and the content.
        let mut strip_top: Option<Pixels> = None;
        let mut strip_bottom: Option<Pixels> = None;
        let pad_y = style.code_block_content_padding_y;
        let mut last_was_delim = true; // pretend "above the block" is a fence
        for sl in shaped {
            // Insert breathing room at fence→content and
            // content→fence transitions.
            if is_code && sl.is_delimiter != last_was_delim {
                if !sl.is_delimiter {
                    // fence → content: top edge of strip lives here
                    strip_top = Some(content_cursor_y);
                }
                content_cursor_y += pad_y;
                if sl.is_delimiter {
                    // content → fence: bottom edge of strip lives here
                    strip_bottom = Some(content_cursor_y);
                }
            }
            // A line carrying math overlays taller than what the
            // body's natural half-leading already covers reserves
            // extra vertical space — `top` above the row's natural
            // top, `bottom` past its natural bottom. The row_height
            // passed to gpui stays at `line_height` so its
            // internal text-vertical-centering keeps the body
            // baseline in the same place across math and non-math
            // rows; we shift `origin.y` down by `extra.top` so the
            // *visible* row top floats up by that amount. Math
            // overlays paint relative to `line.origin.y + ...`,
            // and the formula in `paint_inline_math_overlays`
            // accounts for gpui's centering using `line.row_height`.
            let math_extra = compute_math_row_extra(
                &sl.source_range,
                &inline_math_specs,
                body_ascent,
                body_descent,
                line_height,
            );
            let image_extra =
                compute_image_row_extra(&sl.source_range, &inline_image_specs, line_height);
            let extra = combined_row_extra(math_extra, image_extra);
            let wrap_count = (sl.line.wrap_boundaries().len() as f32) + 1.0;
            let wrapped_h = (line_height + extra.total()) * wrap_count;
            let origin = point(content_left, content_cursor_y + extra.top);
            if !sl.is_delimiter && sl.line.width() > max_content_line_width {
                max_content_line_width = sl.line.width();
            }
            lines.push(LaidOutLine {
                line: sl.line,
                origin,
                row_height: line_height,
                wrapped_height: wrapped_h,
                source_range: sl.source_range,
                display_to_source: sl.display_to_source,
                is_delimiter: sl.is_delimiter,
            });
            content_cursor_y += wrapped_h;
            last_was_delim = sl.is_delimiter;
        }
        // Table pieces: re-base the layout's relative origins into
        // window coordinates. All pieces of one table row share the
        // row's top y; the layout already advanced rows by their
        // tallest cell.
        let mut table_rule_ys: Vec<Pixels> = Vec::new();
        if let Some(layout) = table_layout {
            for piece in layout.pieces {
                lines.push(LaidOutLine {
                    line: piece.line,
                    origin: point(
                        content_left + piece.rel_origin.x,
                        content_top + piece.rel_origin.y,
                    ),
                    row_height: line_height,
                    wrapped_height: piece.wrapped_height,
                    source_range: piece.source_range,
                    display_to_source: piece.display_to_source,
                    is_delimiter: false,
                });
            }
            max_content_line_width = layout.size.width;
            content_cursor_y = content_top + layout.size.height;
            table_rule_ys = layout.rule_ys.iter().map(|ry| content_top + *ry).collect();
        }
        // Trailing content (no closing fence) — extend the strip to
        // cover the trailing breathing room and advance the layout
        // cursor so the block reserves the space.
        if is_code && !last_was_delim {
            content_cursor_y += pad_y;
            strip_bottom = Some(content_cursor_y);
        }

        if lines.is_empty() {
            // Empty block — fabricate a zero-content shaped line so cursor
            // positioning still works on truly empty paragraphs / empty
            // code blocks.
            if let Some(line) = empty_shaped_line(font_size, window) {
                lines.push(LaidOutLine {
                    line,
                    origin: point(content_left, content_cursor_y),
                    row_height: line_height,
                    wrapped_height: line_height,
                    source_range: self.block.source_range.clone(),
                    display_to_source: vec![self.block.source_range.start],
                    is_delimiter: false,
                });
                content_cursor_y += line_height;
            }
        }

        let spacing_below =
            spacing_below_for_block(&self.block.kind, &self.block.containers, &style);
        let extra_below = container_boundary_extra(
            &self.block.containers,
            self.next_containers.as_deref(),
            &style,
        );

        let natural_bottom = content_cursor_y + inner_pad;
        let resolved_content_h =
            (bounds.size.height - extra_above - spacing_above - spacing_below - extra_below)
                .max(natural_bottom - block_top);
        let block_bottom = block_top + resolved_content_h;
        // `block_bounds` covers this leaf's *full* width (including the
        // container-indent strip on the left) so hit-testing routes
        // clicks anywhere on the row to this leaf. The code-block
        // background bounds below shrink to `block_left..block_left +
        // block_width` so the bg only paints over the visible content
        // band, not the indent strip.
        let block_bounds = Bounds::new(
            point(bounds.origin.x, block_top),
            size(bounds.size.width, block_bottom - block_top),
        );

        // Cap horizontal scroll: the rightmost edge of the widest
        // content line should never go further left than
        // `visible_content_width`.
        let max_scroll = (max_content_line_width - visible_content_width).max(px(0.0));
        let scroll_x = scroll_offset.min(max_scroll).max(px(0.0));
        if no_wrap && scroll_x != scroll_offset {
            // Out-of-range cached scroll (e.g. content shrank) — clamp
            // it on the editor so subsequent frames see the corrected
            // value.
            let block_index = self.block_index;
            self.editor.update(cx, |editor, _| {
                editor.set_code_block_scroll(block_index, scroll_x);
            });
        }
        // Only *content* lines translate with horizontal scroll —
        // fence lines stay pinned at `content_left`. The fence rows
        // are a stable frame around the scrolling code area. (Tables
        // have no delimiter lines, so every row translates.)
        if no_wrap && scroll_x > px(0.0) {
            for line in &mut lines {
                if !line.is_delimiter {
                    line.origin.x -= scroll_x;
                }
            }
        }

        let laid_out = LaidOutBlock {
            block_bounds,
            lines,
            source_range: self.block.source_range.clone(),
            embed_ordinal: match &self.block.kind {
                BlockKind::Embed { ordinal } => Some(*ordinal),
                _ => None,
            },
            embed_content: None,
        };

        // Marker-overlay cursor: when the cursor is inside a task
        // item's marker chrome (`- [ ] ` / `- [x] `) the line has
        // those bytes hidden — `local_position_for_source_offset`
        // would map to display column 0 and paint the caret at the
        // content edge. Detect the case here and compute the caret's
        // x position inside the painted overlay glyph instead.
        let marker_caret = compute_marker_overlay_caret(
            &self.block.marker_overlays,
            &self.block.containers,
            &self.block.kind,
            selection.head(),
            &laid_out.lines,
            bounds.origin.x,
            &style,
            window,
        );

        let (mut cursor_quad, selection_quads) = build_caret_and_selection(
            &laid_out,
            selection,
            &style,
            self.is_last_block,
            self.next_block_start,
            marker_caret,
            caret_downstream,
        );

        // Host-supplied highlight washes — the same per-line quad geometry as
        // a selection, in each layer's (fainter, warmer) highlight color. The
        // ranges arrive pre-merged per layer (see
        // `HighlightLayers::merged_by_layer`), so overlapping highlights on one
        // layer paint one wash, never a stacked darker band; layers arrive
        // bottom to top, so an upper layer's wash paints over a lower one.
        let highlight_quads: Vec<TaggedQuad> = highlight_layers
            .iter()
            .flat_map(|(layer, ranges)| {
                build_highlight_quads(&laid_out, ranges, style.highlight_layer_color(*layer))
            })
            .collect();

        if cursor_quad.is_none() && source.is_empty() && self.block_index == 0 {
            // Truly empty document — paint a cursor at the origin so the
            // user sees the editor is focused.
            cursor_quad = Some(TaggedQuad {
                quad: fill(
                    Bounds::new(bounds.origin, size(px(2.0), line_height)),
                    style.caret_color,
                ),
                is_delimiter: false,
            });
        }

        // Code-block paint state. `content_strip` is the darker
        // background band — full block width edge-to-edge so the
        // outer fence-frame bg shows only on the fence rows above
        // and below. `content_clip` is the same rectangle: padding
        // lives *inside* the scroll viewport (CSS-style), so content
        // is free to scroll right up to the bg's edges. The padding
        // shows as visible space at the leading edge when scroll=0
        // and at the trailing edge when scroll=max — it's part of
        // the scrollable inner content, not a clip outside the
        // scroll. `content_left = bounds.origin.x + inner_pad`
        // already places the unscrolled content inset from the left;
        // `max_scroll = max_content_line_width - visible_content_width`
        // already accounts for the symmetric trailing padding at
        // max scroll.
        // Table paint state: scroll geometry plus display mode's
        // hairline rules. Rules sit on the boundary between adjacent
        // rows — the natural half-leading above and below the rule
        // keeps the text breathing. `rules[0]` is the header rule
        // (painted at full rule color; later ones lighter). Edit mode
        // paints no rules — the dimmed delimiter row is the boundary.
        let table_paint = if is_table {
            let mut rules = Vec::new();
            let rule_x = content_left - scroll_x;
            let rule_w = max_content_line_width.max(px(1.0));
            for ry in &table_rule_ys {
                rules.push(Bounds::new(
                    point(rule_x, *ry - px(0.5)),
                    size(rule_w, px(1.0)),
                ));
            }
            Some(TablePaint {
                clip: Bounds::new(
                    point(block_left, block_top),
                    size(block_width, (block_bottom - block_top).max(px(0.0))),
                ),
                content_width: max_content_line_width,
                visible_width: visible_content_width,
                scroll_x,
                rules,
            })
        } else {
            None
        };

        let code_block_paint = if is_code {
            let strip_top = strip_top.unwrap_or(block_top + inner_pad);
            let strip_bottom = strip_bottom.unwrap_or(block_bottom - inner_pad);
            let strip_bounds = Bounds::new(
                point(block_left, strip_top),
                size(block_width, strip_bottom - strip_top),
            );
            // Code-block bg only spans the leaf's visible width
            // (right of any container indent), not the click-area
            // bounds — otherwise the bg would paint over the
            // blockquote indent strip and bury the border bar.
            let bg_bounds = Bounds::new(
                point(block_left, block_top),
                size(block_width, block_bottom - block_top),
            );
            Some(CodeBlockPaint {
                bg_bounds,
                content_strip: strip_bounds,
                content_clip: strip_bounds,
                content_width: max_content_line_width,
                visible_width: visible_content_width,
                scroll_x,
            })
        } else {
            None
        };

        // Display-math in display mode: typeset and stash the
        // result. Typesetting failure (malformed LaTeX) falls
        // through to the regular text-shape path which rendered the
        // raw source bytes — the user gets a fallback view rather
        // than a blank block.
        let math_paint = if matches!(
            self.block.kind,
            BlockKind::DisplayMath {
                edit_mode: false,
                ..
            }
        ) {
            typeset_display_math_block(&self.block, &source).map(|layout| MathPaint {
                layout,
                origin: point(content_left, content_top),
                em_px: font_size,
            })
        } else {
            None
        };

        // Embed block: lay the mapped markdown out inside the quote
        // container and re-base into window coordinates. The regular
        // shaped path above produced only the fully-hidden marker line
        // (the caret / hit-test anchor); this is what paints.
        let embed_paint = if let BlockKind::Embed { ordinal } = &self.block.kind {
            let content = self
                .editor
                .read(cx)
                .state
                .embeds
                .get(*ordinal)
                .map(str::to_string);
            content.map(|content| {
                let inset = style.blockquote_indent;
                let inner_w = (block_width - inset).max(px(1.0));
                let mut layout = layout_embed(&content, &style, inner_w, window);
                let origin = point(content_left + inset, content_top);
                for piece in &mut layout.pieces {
                    piece.rel_origin.x += origin.x;
                    piece.rel_origin.y += origin.y;
                }
                for m in &mut layout.math {
                    m.rel_origin.x += origin.x;
                    m.rel_origin.y += origin.y;
                }
                for (rule, _) in &mut layout.rules {
                    rule.origin.x += origin.x;
                    rule.origin.y += origin.y;
                }
                for bar in &mut layout.bars {
                    bar.origin.x += origin.x;
                    bar.origin.y += origin.y;
                }
                for panel in &mut layout.code_panels {
                    panel.origin.x += origin.x;
                    panel.origin.y += origin.y;
                }
                let content_bounds = Bounds::new(
                    point(block_left, block_top),
                    size(block_width, (block_bottom - block_top).max(px(0.0))),
                );
                let selected = !selection.is_collapsed()
                    && selection.lower_bound() <= self.block.source_range.start
                    && selection.upper_bound() >= self.block.source_range.end;
                EmbedPaint {
                    pieces: layout.pieces,
                    math: layout.math,
                    rules: layout.rules,
                    bars: layout.bars,
                    code_panels: layout.code_panels,
                    bar: Bounds::new(
                        point(block_left + style.blockquote_border_inset, block_top),
                        size(
                            style.blockquote_border_width,
                            (block_bottom - block_top).max(px(0.0)),
                        ),
                    ),
                    bounds: content_bounds,
                    selected,
                }
            })
        } else {
            None
        };

        // Embed caret affinity: the marker's shaped line is fully hidden, so
        // it has zero width and *both* edges of the block map to display
        // column 0 — a caret before and a caret after the embed painted in the
        // same place, which is a lie about where the next character lands.
        // Treat the block the way a caret treats a glyph: leading edge at its
        // left (already the case), trailing edge at its **right**, on the
        // block's last row. Only the x/y move — the caret keeps its own width
        // and row height, so it reads as a caret, not a border.
        if let Some(ep) = embed_paint.as_ref()
            && let Some(caret) = cursor_quad.as_mut()
            && selection.is_collapsed()
            && selection.head() == self.block.source_range.end
        {
            let w = caret.quad.bounds.size.width;
            let h = caret.quad.bounds.size.height;
            caret.quad.bounds.origin = point(
                (ep.bounds.origin.x + ep.bounds.size.width - w).max(ep.bounds.origin.x),
                (ep.bounds.origin.y + ep.bounds.size.height - h).max(ep.bounds.origin.y),
            );
        }

        // Block image in display mode: load and stash bounds. Loading
        // / failed states fall through to the regular shape path
        // (which renders the source bytes as a fallback row).
        let image_paint = if let BlockKind::Image {
            dest_url,
            edit_mode: false,
            ..
        } = &self.block.kind
        {
            match crate::image::load(dest_url, window, cx) {
                crate::image::LoadedImage::Loaded(img) => {
                    let inner_w = (block_width - inner_pad * 2.).max(px(1.0));
                    let size = crate::image::block_size(img.as_ref(), inner_w);
                    Some(ImagePaint {
                        image: img,
                        bounds: Bounds::new(point(content_left, content_top), size),
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        PrepaintState {
            laid_out,
            cursor_quad,
            selection_quads,
            highlight_quads,
            code_block_paint,
            table_paint,
            math_paint,
            inline_math: inline_math_specs,
            image_paint,
            inline_images: inline_image_specs,
            embed_paint,
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
        let (focus_handle, should_register, disabled) = self.editor.update(cx, |editor, _| {
            // Skip IME registration entirely when read-only — a disabled
            // editor accepts no typed text or composition.
            let should = !editor.frame_input_handler_set && !editor.disabled;
            editor.frame_input_handler_set = true;
            (editor.focus_handle.clone(), should, editor.disabled)
        });
        if should_register {
            let editor_bounds = self.editor.read(cx).last_bounds.unwrap_or(bounds);
            window.handle_input(
                &focus_handle,
                ElementInputHandler::new(editor_bounds, self.editor.clone()),
                cx,
            );
        }

        let code_block_paint = prepaint.code_block_paint;

        // Per-level blockquote borders. Painted *first*, so any
        // code-block bg or content paints over them only inside the
        // already-indented content area (the bg is `block_left` to
        // `block_left + block_width`, which starts to the right of
        // every border bar). Each container that's a blockquote
        // contributes one bar at `bounds.origin.x +
        // blockquote_border_inset + i * blockquote_indent`. The bar
        // spans this leaf's full vertical bounds, which now include
        // both `spacing_above` and `spacing_below` (each half a
        // paragraph_gap) — so the bar reads as symmetric around the
        // content rows. When two consecutive leaves share the same
        // blockquote their bars still meet flush at the leaf boundary
        // (`spacing_below` of leaf 1 ends exactly where `spacing_above`
        // of leaf 2 begins).
        // The bar's vertical extent is the layout box minus any
        // container-boundary extras: extras are pure inter-block
        // breathing room and shouldn't be covered by the bar (the bar
        // marks the quoted region, the extras sit outside it). For
        // consecutive blockquoted leaves with the same chain both
        // extras are zero, so bars meet flush at the leaf boundary.
        // Per-level bar extents.
        //
        // The block as a whole reserves `extra_above` / `extra_below`
        // breathing room at any container-boundary transition. But
        // when nesting changes — e.g. block A `[outer]` → block B
        // `[outer, inner]` — the *outer* level is shared between A
        // and B, so the outer bar should stay continuous through the
        // extras while the *inner* bar stops at them. Each level's
        // bar therefore computes its own top / bottom against
        // whether the matching ancestry is present in the immediate
        // neighbor: shared at level L → bar paints flush to the
        // layout box on that side; not shared → bar pulls back by
        // the extra so the boundary breathes.
        let extra_above = container_boundary_extra(
            &self.block.containers,
            self.prev_containers.as_deref(),
            &self.style,
        );
        let extra_below = container_boundary_extra(
            &self.block.containers,
            self.next_containers.as_deref(),
            &self.style,
        );
        for (level, container) in self.block.containers.iter().enumerate() {
            match container {
                Container::BlockQuote { .. } => {
                    let above_continues = level_shared_with_neighbor(
                        level,
                        &self.block.containers,
                        self.prev_containers.as_deref(),
                    );
                    let below_continues = level_shared_with_neighbor(
                        level,
                        &self.block.containers,
                        self.next_containers.as_deref(),
                    );
                    let bar_top = bounds.origin.y
                        + if above_continues {
                            px(0.0)
                        } else {
                            extra_above
                        };
                    let bar_bottom = bounds.origin.y + bounds.size.height
                        - if below_continues {
                            px(0.0)
                        } else {
                            extra_below
                        };
                    let bar_height = (bar_bottom - bar_top).max(px(0.0));
                    let left = bounds.origin.x
                        + container_x_at_level(&self.block.containers, level, &self.style, window)
                        + self.style.blockquote_border_inset;
                    let bar = Bounds::new(
                        point(left, bar_top),
                        size(self.style.blockquote_border_width, bar_height),
                    );
                    window.paint_quad(fill(bar, self.style.blockquote_border_color));
                }
                Container::ListItem { .. } => {
                    // List items have no left-edge chrome of their
                    // own — their visual cue is the marker glyph
                    // shaped into the line, plus the cumulative
                    // `list_indent` already applied by the
                    // container chain.
                }
            }
        }

        // Overlay container markers (cursor-inside `>` glyphs) on top
        // of their level's border bar. Painted *after* the bars and
        // *before* content text so the glyph reads on top of the bar
        // but doesn't compete with body shaping. Because the marker
        // bytes were also pushed into `hidden_ranges`, the shaped
        // line itself starts at `block_left` regardless of whether
        // overlays are present — the user gets the raw `>` they typed
        // without any horizontal shift in the content's position.
        paint_marker_overlays(
            &self.block.marker_overlays,
            &self.block.containers,
            &self.block.kind,
            &prepaint.laid_out.lines,
            bounds.origin.x,
            &self.style,
            window,
            cx,
        );

        // Layered backgrounds for code blocks:
        //
        // 1. Outer rounded fill (`code_block_background`) — full
        //    block. The rounded corners are visible only at the
        //    fence rows, where this bg is uncovered.
        // 2. Inner content strip (`code_block_content_background`) —
        //    full-width band between the fence rows, no rounding.
        //    This is what reads as the "code area".
        if let Some(cb) = &code_block_paint {
            window.paint_quad(quad(
                cb.bg_bounds,
                self.style.code_block_radius,
                self.style.code_block_background,
                Edges::default(),
                gpui::transparent_black(),
                BorderStyle::default(),
            ));
            if cb.content_strip.size.height > px(0.0) {
                window.paint_quad(fill(
                    cb.content_strip,
                    self.style.code_block_content_background,
                ));
            }
        }

        // Thematic break: paint a thin horizontal rule centered on
        // the block's content row. The shaped line (`---` etc.)
        // continues to live in `lines`, but its source bytes are
        // hidden when the cursor is outside the construct (the
        // common case) so the user only sees the rule. When the
        // cursor is on the line the source dims into view in the
        // delimiter color so the user can see what they typed.
        if matches!(self.block.kind, BlockKind::ThematicBreak) {
            let indent = containers_left_indent(&self.block.containers, &self.style, window);
            paint_thematic_break(
                window,
                &prepaint.laid_out.lines,
                bounds.origin.x + indent,
                (bounds.size.width - indent).max(px(1.0)),
                &self.style,
            );
        }

        // Display math (display mode): paint typeset math via the
        // RaTeX adapter. KaTeX fonts register lazily here on first
        // paint — hosting apps may also call
        // `crate::math::register_katex_fonts` at app init.
        if let Some(math_paint) = prepaint.math_paint.take() {
            let _ = crate::math::register_katex_fonts(&window.text_system().clone());
            math_paint.layout.paint(
                math_paint.origin,
                math_paint.em_px,
                self.style.text_color,
                window,
                cx,
            );
        }

        // Block image (display mode): paint the loaded image at the
        // pre-computed bounds. Loading / failed states leave
        // `image_paint == None` and the regular shape path renders
        // the source bytes (or a placeholder row from the empty-
        // block fallback) underneath.
        if let Some(image_paint) = prepaint.image_paint.take() {
            crate::image::paint(image_paint.image, image_paint.bounds, window);
        }

        // Embed block: the quote chrome + the mapped markdown's shaped
        // pieces, clipped to the block bounds (an over-wide embedded code
        // line or table never leaks outside the container). Painted before
        // the regular line pass — the marker's own shaped line is empty, so
        // ordering only matters vs. the caret, which paints last.
        let mut embed_content: Option<EmbedContentGeometry> = None;
        if let Some(ep) = prepaint.embed_paint.take() {
            if ep.selected {
                window.paint_quad(fill(ep.bounds, self.style.selection_color));
            }
            window.paint_quad(fill(ep.bar, self.style.blockquote_border_color));
            let mask = ContentMask { bounds: ep.bounds };
            window.with_content_mask(Some(mask), |window| {
                for panel in &ep.code_panels {
                    window.paint_quad(quad(
                        *panel,
                        self.style.code_block_radius,
                        self.style.code_block_content_background,
                        Edges::default(),
                        gpui::transparent_black(),
                        BorderStyle::default(),
                    ));
                }
                for bar in &ep.bars {
                    window.paint_quad(fill(*bar, self.style.blockquote_border_color));
                }
                for (rule, full) in &ep.rules {
                    let mut color = self.style.table_rule_color;
                    if !full {
                        color.a *= 0.45;
                    }
                    window.paint_quad(fill(*rule, color));
                }
                // Run backgrounds first (inline-code chips), then glyphs —
                // the same two-pass order as the regular content path.
                for piece in &ep.pieces {
                    let _ = piece.line.paint_background(
                        piece.rel_origin,
                        piece.row_height,
                        gpui::TextAlign::Left,
                        None,
                        window,
                        cx,
                    );
                }
                for piece in &ep.pieces {
                    let _ = piece.line.paint(
                        piece.rel_origin,
                        piece.row_height,
                        gpui::TextAlign::Left,
                        None,
                        window,
                        cx,
                    );
                }
                if !ep.math.is_empty() {
                    let _ = crate::math::register_katex_fonts(&window.text_system().clone());
                }
                for m in &ep.math {
                    m.layout
                        .paint(m.rel_origin, m.em_px, self.style.text_color, window, cx);
                }
            });
            // Record what was painted (moving, not cloning — `ep` is dropped
            // here regardless). See `LaidOutBlock::embed_content`.
            embed_content = Some(EmbedContentGeometry {
                pieces: ep
                    .pieces
                    .into_iter()
                    .map(|p| (p.line.text.clone(), p.rel_origin))
                    .collect(),
                code_panels: ep.code_panels,
                math_origins: ep.math.into_iter().map(|m| m.rel_origin).collect(),
                rules: ep.rules.into_iter().map(|(r, _)| r).collect(),
                bars: ep.bars,
            });
        }

        let cursor_quad = prepaint.cursor_quad.take();
        let selection_quads = std::mem::take(&mut prepaint.selection_quads);
        let highlight_quads = std::mem::take(&mut prepaint.highlight_quads);
        let focused = focus_handle.is_focused(window);

        // Split paint into delimiter and content phases. Fence-row
        // text and the cursor / selection quads attached to fence
        // rows paint *outside* the content mask so they stay visible
        // regardless of horizontal scroll (fences are pinned in x).
        // Content text and content-row cursor / selection paint
        // *inside* the mask so long lines clip at the visible band's
        // right edge. Non-code blocks have `code_block_paint == None`
        // and skip the mask entirely.
        if let Some(cb) = &code_block_paint {
            let (delim_hl, content_hl): (Vec<_>, Vec<_>) =
                highlight_quads.into_iter().partition(|t| t.is_delimiter);
            let (delim_sel, content_sel): (Vec<_>, Vec<_>) =
                selection_quads.into_iter().partition(|t| t.is_delimiter);
            let (cursor_for_delim, cursor_for_content) = match cursor_quad {
                Some(tq) if tq.is_delimiter => (Some(tq), None),
                Some(tq) => (None, Some(tq)),
                None => (None, None),
            };

            // Phase 1: delimiter lines + their cursor / selection,
            // unmasked.
            for tq in delim_hl {
                window.paint_quad(tq.quad);
            }
            for tq in delim_sel {
                window.paint_quad(tq.quad);
            }
            for laid in &prepaint.laid_out.lines {
                if laid.is_delimiter {
                    let _ = laid.line.paint(
                        laid.origin,
                        laid.row_height,
                        gpui::TextAlign::Left,
                        None,
                        window,
                        cx,
                    );
                }
            }
            if focused
                && !disabled
                && let Some(tq) = cursor_for_delim
            {
                window.paint_quad(tq.quad);
            }

            // Phase 2: content lines + their cursor / selection,
            // masked to the content strip.
            let mask = ContentMask {
                bounds: cb.content_clip,
            };
            window.with_content_mask(Some(mask), |window| {
                for tq in content_hl {
                    window.paint_quad(tq.quad);
                }
                for tq in content_sel {
                    window.paint_quad(tq.quad);
                }
                for laid in &prepaint.laid_out.lines {
                    if !laid.is_delimiter {
                        let _ = laid.line.paint(
                            laid.origin,
                            laid.row_height,
                            gpui::TextAlign::Left,
                            None,
                            window,
                            cx,
                        );
                    }
                }
                if focused
                    && !disabled
                    && let Some(tq) = cursor_for_content
                {
                    window.paint_quad(tq.quad);
                }
            });
        } else {
            // Non-code blocks: no fence split. Tables additionally
            // clip to their visible band when wider than the content
            // area (they never soft-wrap — the same horizontal-scroll
            // treatment as code blocks, minus the background chrome)
            // and paint their hairline rules under everything.
            let table_mask = prepaint.table_paint.as_ref().and_then(|tp| {
                (tp.content_width > tp.visible_width).then_some(ContentMask { bounds: tp.clip })
            });
            let table_rules: Vec<(Bounds<Pixels>, bool)> = prepaint
                .table_paint
                .as_ref()
                .map(|tp| {
                    tp.rules
                        .iter()
                        .enumerate()
                        .map(|(i, r)| (*r, i == 0))
                        .collect()
                })
                .unwrap_or_default();
            window.with_content_mask(table_mask, |window| {
                // Table rules paint first — chrome under content.
                for (rule, is_header_rule) in &table_rules {
                    let mut color = self.style.table_rule_color;
                    if !is_header_rule {
                        color.a *= 0.45;
                    }
                    window.paint_quad(fill(*rule, color));
                }
                // Run backgrounds (the inline-code chip fill carried on
                // `TextRun::background_color`) paint first so the
                // selection wash and the glyphs layer on top of them.
                // `WrappedLine::paint` itself only draws glyphs +
                // underline/strikethrough — backgrounds need the explicit
                // `paint_background` pass or the chip never shows.
                for laid in &prepaint.laid_out.lines {
                    let _ = laid.line.paint_background(
                        laid.origin,
                        laid.row_height,
                        gpui::TextAlign::Left,
                        None,
                        window,
                        cx,
                    );
                }
                for tq in highlight_quads {
                    window.paint_quad(tq.quad);
                }
                for tq in selection_quads {
                    window.paint_quad(tq.quad);
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
                // Inline math overlays paint *after* the line text (so
                // they sit on top of the NBSP placeholders that
                // reserved the space) but *before* the cursor (so the
                // caret remains visible above the math).
                paint_inline_math_overlays(
                    &prepaint.inline_math,
                    &prepaint.laid_out.lines,
                    &self.style,
                    &self.block.kind,
                    window,
                    cx,
                );
                // Inline image overlays — same paint ordering as math
                // (over the line text, under the cursor). Each spec
                // paints exactly where its substitution placed pad
                // glyphs in the shaped line.
                paint_inline_image_overlays(
                    &prepaint.inline_images,
                    &prepaint.laid_out.lines,
                    window,
                );
                if focused
                    && !disabled
                    && let Some(tq) = cursor_quad
                {
                    window.paint_quad(tq.quad);
                }
            });
        }

        // Horizontal scrollbar overlay — only when the code block has
        // overflow. A thin track at the bottom of the bg fill, with a
        // muted thumb whose size and offset reflect the visible /
        // total ratio. Painted *outside* the content mask so it stays
        // visible when the user is scrolled past the right edge of
        // their content.
        if let Some(cb) = &code_block_paint
            && cb.content_width > cb.visible_width
        {
            paint_horizontal_scrollbar(window, cb, &self.style);
        }
        // Table scrollbar: same overlay as code blocks, tracked along
        // the block's bottom edge (tables have no fence seam to sit on).
        if let Some(tp) = &prepaint.table_paint
            && tp.content_width > tp.visible_width
        {
            paint_h_scrollbar_at(
                window,
                &self.style,
                tp.clip.bottom() - px(3.0),
                tp.clip.left(),
                tp.clip.right(),
                tp.content_width,
                tp.visible_width,
                tp.scroll_x,
            );
        }

        // Horizontal scroll-wheel routing for the no-wrap blocks
        // (code + table): one listener over the block's scrollable
        // area, writing the per-block scroll slot.
        let scroll_wheel_area = if let Some(cb) = code_block_paint {
            Some((
                cb.bg_bounds,
                (cb.content_width - cb.visible_width).max(px(0.0)),
            ))
        } else {
            prepaint
                .table_paint
                .as_ref()
                .map(|tp| (tp.clip, (tp.content_width - tp.visible_width).max(px(0.0))))
        };
        if let Some((bg_bounds, max_scroll)) = scroll_wheel_area {
            // Capture for the scroll-wheel listener.
            let editor = self.editor.clone();
            let block_index = self.block_index;
            window.on_mouse_event(move |event: &ScrollWheelEvent, phase, _window, cx| {
                if !phase.bubble() {
                    return;
                }
                if !bg_bounds.contains(&event.position) {
                    return;
                }
                let dx = match event.delta {
                    // Trackpad / pixel-precision wheel.
                    ScrollDelta::Pixels(p) => p.x,
                    // Discrete wheel — scale lines into ~16 px/line
                    // for predictable feel.
                    ScrollDelta::Lines(p) => px(p.x * 16.0),
                };
                if dx == px(0.0) {
                    return;
                }
                editor.update(cx, |editor, cx| {
                    let prev = editor.code_block_scroll(block_index);
                    // Scroll wheel deltas on macOS are inverted (a
                    // swipe-left on the trackpad gives positive dx
                    // and means "scroll right" — i.e. show content
                    // further to the right). Subtract `dx` so the
                    // content tracks the swipe direction.
                    let next = (prev - dx).clamp(px(0.0), max_scroll);
                    if next != prev {
                        editor.set_code_block_scroll(block_index, next);
                        cx.notify();
                    }
                });
            });
        }

        let laid_out = std::mem::replace(
            &mut prepaint.laid_out,
            LaidOutBlock {
                block_bounds: Bounds::default(),
                lines: Vec::new(),
                source_range: 0..0,
                embed_ordinal: None,
                embed_content: None,
            },
        );
        let laid_out = LaidOutBlock {
            embed_content,
            ..laid_out
        };
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

/// Paint each container marker as a glyph in its level's indent
/// strip. Dispatches on the container kind at the marker's level:
///
/// * `BlockQuote` — paint a `>` glyph centered on the level's left
///   border bar so the bar passes through the glyph's middle.
/// * `ListItem` — paint the item's marker text (`• `, `- `, or
///   `1. `) right-aligned within the item's indent strip so every
///   sibling's content edge lines up regardless of marker width.
///
/// The marker's source byte selects which shaped line provides the y
/// origin, so multi-line leaves (soft-wrap, hard-break continuations)
/// pick the right row. Painted in the body font + delimiter color
/// so the chrome reads cohesive with the rest of the indent column.
#[allow(clippy::too_many_arguments)]
fn paint_marker_overlays(
    overlays: &[crate::render_spec::MarkerOverlay],
    containers: &[Container],
    kind: &BlockKind,
    lines: &[LaidOutLine],
    block_origin_x: Pixels,
    style: &MarkdownStyle,
    window: &mut Window,
    cx: &mut App,
) {
    if overlays.is_empty() {
        return;
    }
    for marker in overlays {
        let Some(level_container) = containers.get(marker.level).cloned() else {
            continue;
        };
        let Some(line) = lines.iter().find(|l| {
            l.source_range.start <= marker.source_range.start
                && marker.source_range.start < l.source_range.end
        }) else {
            continue;
        };
        match level_container {
            Container::BlockQuote { .. } => {
                paint_blockquote_marker_overlay(
                    containers,
                    marker.level,
                    line,
                    block_origin_x,
                    kind,
                    style,
                    window,
                    cx,
                );
            }
            Container::ListItem {
                kind: item_kind,
                cursor_inside,
                ..
            } => {
                paint_list_marker_overlay(
                    containers,
                    marker.level,
                    line,
                    block_origin_x,
                    item_kind,
                    cursor_inside,
                    style,
                    window,
                    cx,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_blockquote_marker_overlay(
    containers: &[Container],
    level: usize,
    line: &LaidOutLine,
    block_origin_x: Pixels,
    kind: &BlockKind,
    style: &MarkdownStyle,
    window: &mut Window,
    cx: &mut App,
) {
    let font_size = font_size_for_block(kind, style);
    let font = base_font_for_block(kind, style);
    let runs = [TextRun {
        len: 1,
        font,
        color: style.delimiter_color,
        background_color: None,
        underline: None,
        strikethrough: None,
    }];
    let Some(glyph) = window
        .text_system()
        .shape_text(SharedString::from(">"), font_size, &runs, None, None)
        .ok()
        .and_then(|mut v| v.drain(..).next())
    else {
        return;
    };
    let bar_left = block_origin_x
        + container_x_at_level(containers, level, style, window)
        + style.blockquote_border_inset;
    // Center the glyph horizontally on the bar so the bar passes
    // through the glyph's middle. The glyph is wider than the bar,
    // so its left/right edges spill into the indent on both sides —
    // that's intentional, the marker reads as *integrated* with the
    // bar rather than floating beside it.
    let glyph_x = bar_left + (style.blockquote_border_width - glyph.width()) / 2.0;
    let _ = glyph.paint(
        point(glyph_x, line.origin.y),
        line.row_height,
        gpui::TextAlign::Left,
        None,
        window,
        cx,
    );
}

#[allow(clippy::too_many_arguments)]
fn paint_list_marker_overlay(
    containers: &[Container],
    level: usize,
    line: &LaidOutLine,
    block_origin_x: Pixels,
    kind: ListItemKind,
    cursor_inside: bool,
    style: &MarkdownStyle,
    window: &mut Window,
    cx: &mut App,
) {
    let display = list_marker_display_text(kind, cursor_inside);
    let Some(glyph) = shape_marker_text(&display, style, window) else {
        return;
    };
    // Right-align the marker glyph against the level's *content
    // edge* (the X where the next level — or the leaf body — starts).
    // The marker text already includes its trailing space, so the
    // glyph's right edge sitting at the content edge gives natural
    // separation from the body. Items in the same list with shorter
    // markers (e.g. `1. ` next to `28. `) right-align under a common
    // content edge — periods and bullets line up.
    let strip_right = block_origin_x + container_x_at_level(containers, level + 1, style, window);
    let glyph_x = strip_right - glyph.width();
    let _ = glyph.paint(
        point(glyph_x, line.origin.y),
        line.row_height,
        gpui::TextAlign::Left,
        None,
        window,
        cx,
    );
}

/// Compute the cursor caret quad for a cursor that lands inside a
/// task-item's marker chrome (`- [ ] ` / `- [x] `).
///
/// When the cursor's source offset sits inside a list item's
/// `marker_overlays[i].source_range`, the line has those bytes
/// hidden — the regular `display_to_source` path would map every
/// hidden byte to display column 0, painting the caret at the
/// content edge regardless of which marker byte the cursor is on.
/// That's wrong for task items, where the user navigates *into* the
/// brackets to toggle the checkbox: they need to see their caret
/// inside the brackets.
///
/// This helper detects the case by:
///   1. finding a marker overlay whose source_range contains the
///      cursor offset;
///   2. checking that the corresponding container is a task-list
///      `ListItem` with `cursor_inside == true` (so the overlay is
///      currently rendering raw `- [ ] ` chrome);
///   3. shaping the overlay text in the marker font, computing the
///      column inside it for `cursor_offset - source_range.start`,
///      and converting that to an absolute pixel x position.
///
/// Returns `None` for cursors that don't match. The bullet-only
/// overlay path (non-task unordered) bypasses this because bytes
/// 0..2 are forbidden by `is_list_indent_interior` — the cursor
/// never lands there.
#[allow(clippy::too_many_arguments)]
fn compute_marker_overlay_caret(
    overlays: &[crate::render_spec::MarkerOverlay],
    containers: &[Container],
    kind: &BlockKind,
    cursor_offset: usize,
    lines: &[LaidOutLine],
    block_origin_x: Pixels,
    style: &MarkdownStyle,
    window: &mut Window,
) -> Option<TaggedQuad> {
    for overlay in overlays {
        if cursor_offset < overlay.source_range.start || cursor_offset > overlay.source_range.end {
            continue;
        }
        let level_container = containers.get(overlay.level)?.clone();
        let (item_kind, cursor_inside) = match level_container {
            Container::ListItem {
                kind,
                cursor_inside,
                ..
            } => (kind, cursor_inside),
            // Blockquote markers don't host a cursor — bytes are
            // forbidden positions, so the cursor never lands there.
            Container::BlockQuote { .. } => continue,
        };
        // Only task items currently render an editable raw form
        // when cursor_inside; the cursor wouldn't land on bullet
        // bytes for plain unordered items because they're
        // forbidden positions.
        if !matches!(item_kind, ListItemKind::Unordered(_, Some(_))) || !cursor_inside {
            continue;
        }
        let display = list_marker_display_text(item_kind, cursor_inside);
        let glyph = shape_marker_text(&display, style, window)?;
        let line = lines.iter().find(|l| {
            l.source_range.start <= overlay.source_range.start
                && overlay.source_range.start < l.source_range.end
        })?;
        // The overlay paints right-aligned ending at the level's
        // content edge. Mirror `paint_list_marker_overlay`'s x
        // calculation so the caret lands inside the painted glyph.
        let strip_right =
            block_origin_x + container_x_at_level(containers, overlay.level + 1, style, window);
        let glyph_left = strip_right - glyph.width();
        let byte_in_overlay = cursor_offset - overlay.source_range.start;
        let local = glyph
            .position_for_index(byte_in_overlay, line.row_height)
            .unwrap_or_else(|| point(px(0.0), px(0.0)));
        let caret_x = glyph_left + local.x;
        let caret_y = line.origin.y + local.y;
        return Some(TaggedQuad {
            quad: fill(
                Bounds::new(point(caret_x, caret_y), size(px(2.0), line.row_height)),
                style.caret_color,
            ),
            is_delimiter: line.is_delimiter,
        });
    }
    let _ = kind;
    None
}

/// Paint each inline math overlay over the placeholder run that
/// reserved its horizontal space. For each overlay:
///
/// 1. Find the laid-out line whose source range contains the
///    overlay's `source_range.start` (a math construct never spans
///    multiple shaped lines because `augment_block_with_math`
///    substitutes the whole construct with non-breaking-padding
///    that wraps as a single block).
/// 2. Walk the line's `display_to_source` map to find the display
///    byte offset where the substitution begins.
/// 3. Convert that display offset to a pixel `x` via
///    `WrappedLine::position_for_index`.
/// 4. Compute the `y` so the math's baseline aligns with the line's
///    text baseline. We resolve the body font and read its actual
///    ascent from gpui's text system rather than approximating —
///    `0.78 * row_height` was close on Newsreader-like serifs but
///    visibly off elsewhere.
fn paint_inline_math_overlays(
    specs: &[InlineMathPaintSpec],
    lines: &[LaidOutLine],
    style: &MarkdownStyle,
    block_kind: &BlockKind,
    window: &mut Window,
    cx: &mut App,
) {
    if specs.is_empty() {
        return;
    }
    let body_font = base_font_for_block(block_kind, style);
    let body_font_id = window.text_system().resolve_font(&body_font);
    for spec in specs {
        let Some(line) = lines.iter().find(|l| {
            l.source_range.start <= spec.source_range.start
                && spec.source_range.start < l.source_range.end
        }) else {
            continue;
        };
        // Locate the display offset where this substitution starts.
        // `display_to_source` is dense — one entry per display byte
        // plus a sentinel — so the first index whose mapped source
        // == spec.source_range.start is the substitution's leading
        // edge.
        let display_offset = line
            .display_to_source
            .iter()
            .position(|&src| src == spec.source_range.start);
        let Some(display_offset) = display_offset else {
            continue;
        };
        let local = match line
            .line
            .position_for_index(display_offset, line.row_height)
        {
            Some(p) => p,
            None => continue,
        };
        let math_left = line.origin.x + local.x;
        // gpui's `WrappedLine::paint` vertically *centers* the
        // shaped text within the row_height: the actual baseline
        // lands at `origin.y + (row_height - text_height)/2 +
        // ascent`. We mirror that math here so the math overlay
        // sits on the same baseline gpui drew the surrounding text
        // on. Skipping the half-leading offset (using just
        // `origin.y + ascent`) was what the user spotted as math
        // sitting too high in body-text rows.
        let body_ascent = window.text_system().ascent(body_font_id, spec.em_px);
        let body_descent = window.text_system().descent(body_font_id, spec.em_px);
        let text_height = body_ascent + body_descent;
        let half_leading = ((line.row_height - text_height) / 2.0).max(px(0.0));
        let text_baseline = line.origin.y + local.y + half_leading + body_ascent;
        let math_baseline = spec.layout.baseline(spec.em_px);
        let math_top = text_baseline - math_baseline;
        spec.layout.paint(
            point(math_left, math_top),
            spec.em_px,
            style.text_color,
            window,
            cx,
        );
        let _ = spec.display_style; // currently unused; retained for future styling
    }
}

/// Paint each inline-image overlay on top of the shaped line that
/// hosts its substitution. Mirrors [`paint_inline_math_overlays`]
/// — the substitution display offset gives the x position; the
/// image is vertically centered on the row (CSS
/// `vertical-align: middle` style) so it composes naturally with
/// surrounding text. The row-height extras computed in
/// [`compute_image_row_extra`] guarantee the row is tall enough for
/// any image overshoot, so the centered image always fits.
fn paint_inline_image_overlays(
    specs: &[InlineImagePaintSpec],
    lines: &[LaidOutLine],
    window: &mut Window,
) {
    if specs.is_empty() {
        return;
    }
    for spec in specs {
        let Some(line) = lines.iter().find(|l| {
            l.source_range.start <= spec.source_range.start
                && spec.source_range.start < l.source_range.end
        }) else {
            continue;
        };
        let display_offset = line
            .display_to_source
            .iter()
            .position(|&src| src == spec.source_range.start);
        let Some(display_offset) = display_offset else {
            continue;
        };
        let local = match line
            .line
            .position_for_index(display_offset, line.row_height)
        {
            Some(p) => p,
            None => continue,
        };
        let image_left = line.origin.x + local.x;
        // Vertically center the image on the row. `line.origin.y`
        // already accounts for any row-extra reservation
        // (`compute_image_row_extra`) so centering inside
        // `row_height` gives a row-symmetric position even when
        // the image is taller than the body font.
        let image_top =
            line.origin.y + local.y + (line.row_height - spec.display_size.height) / 2.0;
        let bounds = Bounds::new(point(image_left, image_top), spec.display_size);
        crate::image::paint(spec.image.clone(), bounds, window);
    }
}

/// Paint the rule for a `BlockKind::ThematicBreak` — a thin
/// horizontal line centered vertically on the block's content row.
/// The block's `lines` list always contains exactly one shaped line
/// (the source bytes, possibly hidden); its `origin.y` plus half its
/// `row_height` gives the vertical center for the rule.
fn paint_thematic_break(
    window: &mut Window,
    lines: &[LaidOutLine],
    left: Pixels,
    width: Pixels,
    style: &MarkdownStyle,
) {
    let Some(line) = lines.first() else {
        return;
    };
    let center_y = line.origin.y + line.row_height / 2.0;
    let half_thickness = style.thematic_break_thickness / 2.0;
    let bar = Bounds::from_corners(
        point(left, center_y - half_thickness),
        point(left + width, center_y + half_thickness),
    );
    window.paint_quad(fill(bar, style.thematic_break_color));
}

fn paint_horizontal_scrollbar(window: &mut Window, cb: &CodeBlockPaint, style: &MarkdownStyle) {
    // Sit on the seam between the content strip and the closing fence
    // row — the strip's bottom edge, so the bar occupies the top
    // sliver of the closing fence row: visually adjacent to the
    // scrollable region without overlapping the content baseline.
    // Track alignment follows the content text (inset by
    // `inner_pad`), not the dark bg's edges.
    paint_h_scrollbar_at(
        window,
        style,
        cb.content_strip.bottom(),
        cb.content_clip.left(),
        cb.content_clip.right(),
        cb.content_width,
        cb.visible_width,
        cb.scroll_x,
    );
}

/// Shared horizontal-scrollbar painter for the no-wrap blocks (code
/// blocks and tables). `track_y` is the top edge of the 3px track.
#[allow(clippy::too_many_arguments)]
fn paint_h_scrollbar_at(
    window: &mut Window,
    style: &MarkdownStyle,
    track_y: Pixels,
    track_left: Pixels,
    track_right: Pixels,
    content_width: Pixels,
    visible_width: Pixels,
    scroll_x: Pixels,
) {
    let track_h = px(3.0);
    let track_w = (track_right - track_left).max(px(1.0));

    // Thumb: proportional to visible / content, offset by scroll
    // position. Minimum thumb width keeps it draggable-feeling at
    // very long content.
    let ratio = (visible_width / content_width).clamp(0.05, 1.0);
    let thumb_w = (track_w * ratio).max(px(24.0));
    let scroll_ratio = if content_width > visible_width {
        f32::from(scroll_x) / f32::from(content_width - visible_width)
    } else {
        0.0
    };
    let thumb_x = track_left + (track_w - thumb_w) * scroll_ratio;

    // Use the delimiter color (theme.muted_foreground) at low alpha so
    // the thumb reads as chrome, not content.
    let mut thumb_color = style.delimiter_color;
    thumb_color.a *= 0.5;

    window.paint_quad(quad(
        Bounds::from_corners(
            point(thumb_x, track_y),
            point(thumb_x + thumb_w, track_y + track_h),
        ),
        track_h / 2.,
        thumb_color,
        Edges::default(),
        gpui::transparent_black(),
        BorderStyle::default(),
    ));
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
        BlockKind::Paragraph
        | BlockKind::ThematicBreak
        | BlockKind::Image { .. }
        | BlockKind::Embed { .. }
        | BlockKind::Table { .. } => style.font_size,
        BlockKind::CodeBlock { .. } => style.mono_font_size,
        // Display math: in edit mode shapes mono LaTeX; in display
        // mode the shaped-text path produces a single zero-width
        // line and the math overlay paints on top — either way the
        // base font size is the mono one, matching how an inline
        // code span renders.
        BlockKind::DisplayMath { .. } => style.mono_font_size,
    }
}

/// Half of the block's vertical breathing room. Each block reserves
/// the *same* amount above and below its content (see
/// `spacing_below_for_block`) so the inter-block gap between two
/// blocks of the same kind is preserved (`above + below`), and so any
/// per-block decoration spanning the full layout bounds — most
/// importantly the blockquote border bar — sits symmetric around the
/// content rows. With the previous "all above" model the bar extended
/// roughly a `paragraph_gap` above the text top and stopped at the
/// text bottom, which read as visually unbalanced.
///
/// Blocks inside a list item (`containers` carries a
/// `Container::ListItem`) tighten the non-heading factor by
/// `style.list_item_gap_factor`: list items should read as lines
/// within a block, not as full paragraphs. The full `paragraph_gap`
/// re-appears at the list ↔ neighbor boundary because the non-list
/// neighbor contributes its own untightened half plus the
/// `container_boundary_extra` both sides add when chains differ.
fn spacing_above_for_block(
    kind: &BlockKind,
    containers: &[Container],
    style: &MarkdownStyle,
) -> Pixels {
    let in_list_item = containers
        .iter()
        .any(|c| matches!(c, Container::ListItem { .. }));
    let rems_factor = match kind {
        BlockKind::Heading { level } if *level <= 2 => 1.5,
        BlockKind::Heading { .. } => 1.25,
        BlockKind::Paragraph
        | BlockKind::CodeBlock { .. }
        | BlockKind::ThematicBreak
        | BlockKind::DisplayMath { .. }
        | BlockKind::Image { .. }
        | BlockKind::Embed { .. }
        | BlockKind::Table { .. } => {
            if in_list_item {
                style.paragraph_gap.0 * style.list_item_gap_factor
            } else {
                style.paragraph_gap.0
            }
        }
    };
    px(f32::from(style.font_size) * rems_factor / 2.0)
}

/// Symmetric companion to `spacing_above_for_block`. Stacking two
/// blocks of the same kind reproduces the old single-`paragraph_gap`
/// inter-block gap (`above_2 + below_1 = 2 * (factor / 2) = factor`).
/// Mixed-kind transitions (e.g. paragraph → heading) average the two
/// factors instead of using just the next block's, which slightly
/// smooths the visual rhythm without disturbing same-kind sequences.
fn spacing_below_for_block(
    kind: &BlockKind,
    containers: &[Container],
    style: &MarkdownStyle,
) -> Pixels {
    spacing_above_for_block(kind, containers, style)
}

/// Structural equality for two chain entries — ignores
/// `cursor_inside`, which is purely a focus-state flag and must not
/// affect inter-block layout. Without this discriminator,
/// `container_boundary_extra` and `level_shared_with_neighbor` would
/// compare two list items in the same list as "different chains"
/// whenever the cursor moved between them, injecting / removing a
/// `container_boundary_gap` half on the boundary and shifting the
/// list's vertical spacing every keystroke.
fn containers_match_structurally(a: &Container, b: &Container) -> bool {
    match (a, b) {
        (Container::BlockQuote { .. }, Container::BlockQuote { .. }) => true,
        (
            Container::ListItem {
                kind: ka,
                marker_byte_len: ma,
                list_max_marker_text: la,
                ..
            },
            Container::ListItem {
                kind: kb,
                marker_byte_len: mb,
                list_max_marker_text: lb,
                ..
            },
        ) => ka == kb && ma == mb && la == lb,
        _ => false,
    }
}

fn chains_match_structurally(a: &[Container], b: &[Container]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| containers_match_structurally(x, y))
}

/// Extra breathing room added on the side of `containers` that faces
/// `neighbor` when the two chains differ (paragraph → blockquote, or
/// the move into / out of a nested level). Splits
/// `style.container_boundary_gap` half-and-half between the two
/// adjacent blocks so the total extra at the boundary is one full
/// `container_boundary_gap`. Zero when chains match (consecutive
/// leaves of the same blockquote get no extra) and zero at doc
/// start / end (`neighbor == None`) so the editor's leading and
/// trailing edges stay flush against the editor frame as before.
fn container_boundary_extra(
    containers: &[Container],
    neighbor: Option<&[Container]>,
    style: &MarkdownStyle,
) -> Pixels {
    let Some(neighbor) = neighbor else {
        return px(0.0);
    };
    if chains_match_structurally(neighbor, containers) {
        return px(0.0);
    }
    px(f32::from(style.font_size) * style.container_boundary_gap.0 / 2.0)
}

/// Does `neighbor`'s chain match `containers` up to *and including*
/// `level`? Used by per-level bar painting to decide whether the bar
/// for level L should extend through the block's boundary extras: if
/// the same ancestry (outermost → level L) is present on the other
/// side, the bar at level L paints flush to the layout box and joins
/// the neighbor's bar at the same level; otherwise it pulls back by
/// the extra so the new (or removed) level reads as starting / ending
/// inside the breathing room.
fn level_shared_with_neighbor(
    level: usize,
    containers: &[Container],
    neighbor: Option<&[Container]>,
) -> bool {
    let Some(neighbor) = neighbor else {
        return false;
    };
    if neighbor.len() <= level || containers.len() <= level {
        return false;
    }
    containers[..=level]
        .iter()
        .zip(neighbor[..=level].iter())
        .all(|(a, b)| containers_match_structurally(a, b))
}

/// Inner padding for a block's visible region — non-zero only for
/// code blocks, which inset their content from the background fill.
fn block_inner_padding(kind: &BlockKind, style: &MarkdownStyle) -> Pixels {
    match kind {
        BlockKind::CodeBlock { .. } => style.code_block_padding,
        _ => px(0.0),
    }
}

fn is_code_block(kind: &BlockKind) -> bool {
    matches!(kind, BlockKind::CodeBlock { .. })
}

fn is_table_block(kind: &BlockKind) -> bool {
    matches!(kind, BlockKind::Table { .. })
}

/// Cumulative left indent contributed by every container (blockquote,
/// list-item) that wraps this leaf. The leaf's content starts
/// `containers_left_indent` inset from `bounds.origin.x`; a per-level
/// decoration painted at level L (e.g. a blockquote border bar) sits
/// at `bounds.origin.x + container_x_at_level(.., L, ..)`, *before*
/// L's own indent contribution.
fn containers_left_indent(
    containers: &[Container],
    style: &MarkdownStyle,
    window: &mut Window,
) -> Pixels {
    container_x_at_level(containers, containers.len(), style, window)
}

/// Cumulative left indent contributed by `containers[..up_to]` — i.e.
/// the X offset (relative to the layout box) of the *content edge* of
/// the container at index `up_to`. With `up_to == containers.len()`
/// this matches `containers_left_indent`. Used when a per-level
/// decoration (a blockquote border bar at level L) needs to sit in
/// the gap *before* its own indent contribution: pass `up_to = L` to
/// get the X of that gap's left edge.
///
/// Each container kind contributes its own indent:
///
/// * `BlockQuote` — fixed `style.blockquote_indent` per level.
/// * `ListItem` — `style.list_indent` of leading padding plus the
///   shaped pixel width of the parent list's widest marker
///   (`list_max_marker_text`). This is what makes every item in the
///   list align at the same content edge regardless of its own
///   marker's width: a `1.` and a `24.` in the same list both
///   resolve to the same indent because they share the same
///   `list_max_marker_text`. The marker glyph itself paints as an
///   overlay inside this strip — see `paint_marker_overlays`.
fn container_x_at_level(
    containers: &[Container],
    up_to: usize,
    style: &MarkdownStyle,
    window: &mut Window,
) -> Pixels {
    let limit = up_to.min(containers.len());
    let mut acc = px(0.0);
    for c in &containers[..limit] {
        match c {
            Container::BlockQuote { .. } => acc += style.blockquote_indent,
            Container::ListItem {
                list_max_marker_text,
                ..
            } => {
                let marker_w = measure_marker_text_width(list_max_marker_text, style, window);
                acc += style.list_indent + marker_w;
            }
        }
    }
    acc
}

/// Pixel width budget for an indent-defining marker text shaped in
/// the body font at the body font size. Two cases:
///
/// * **Ordered marker** (`text` starts with a digit run, e.g.
///   `"11. "`) — reserve `max(shape("0".."9"))` pixels per digit
///   plus the shaped suffix. The renderer hands us a string built
///   from `start + count - 1`; that gives the *digit count*
///   accurately but doesn't know the font's per-digit shape
///   widths. By widening every digit slot to the worst-case glyph
///   we cover the user's case where `24.` may shape wider than
///   `31.` in a proportional-figure font, without needing to
///   re-shape every actual marker in the list.
///
/// * **Other text** (unordered bullet, anything else) — shape and
///   measure directly.
fn measure_marker_text_width(text: &str, style: &MarkdownStyle, window: &mut Window) -> Pixels {
    if text.is_empty() {
        return px(0.0);
    }
    if let Some((digit_count, suffix)) = split_digit_prefix(text) {
        let widest_digit = widest_single_digit_width(style, window);
        let suffix_w = if suffix.is_empty() {
            px(0.0)
        } else {
            shape_marker_text(suffix, style, window)
                .map(|line| line.width())
                .unwrap_or(px(0.0))
        };
        return widest_digit * (digit_count as f32) + suffix_w;
    }
    shape_marker_text(text, style, window)
        .map(|line| line.width())
        .unwrap_or(px(0.0))
}

/// Split `text` into `(digit_count, suffix)` if it begins with one
/// or more ASCII digits. `None` for non-numeric markers (so callers
/// can fall back to direct shaping).
fn split_digit_prefix(text: &str) -> Option<(usize, &str)> {
    let bytes = text.as_bytes();
    let mut n = 0;
    while n < bytes.len() && bytes[n].is_ascii_digit() {
        n += 1;
    }
    if n == 0 {
        return None;
    }
    Some((n, &text[n..]))
}

/// Pixel width of the widest single ASCII digit (0-9) shaped in the
/// body font. Stable per font + size, but recomputed per call —
/// `shape_text` is internally cached by gpui so the redundant calls
/// are cheap. Centralizing here keeps `measure_marker_text_width`
/// readable.
fn widest_single_digit_width(style: &MarkdownStyle, window: &mut Window) -> Pixels {
    let mut max_w = px(0.0);
    for d in b'0'..=b'9' {
        let s = (d as char).to_string();
        let w = shape_marker_text(&s, style, window)
            .map(|line| line.width())
            .unwrap_or(px(0.0));
        if w > max_w {
            max_w = w;
        }
    }
    max_w
}

/// Shape `text` in the body font + delimiter color so it can be
/// painted as a marker overlay or measured for indent.
fn shape_marker_text(
    text: &str,
    style: &MarkdownStyle,
    window: &mut Window,
) -> Option<Arc<WrappedLine>> {
    if text.is_empty() {
        return None;
    }
    let font = gpui::Font {
        family: style.font_family.clone(),
        features: gpui::FontFeatures::default(),
        fallbacks: None,
        weight: FontWeight::NORMAL,
        style: FontStyle::Normal,
    };
    let runs = [TextRun {
        len: text.len(),
        font,
        color: style.delimiter_color,
        background_color: None,
        underline: None,
        strikethrough: None,
    }];
    window
        .text_system()
        .shape_text(
            SharedString::from(text.to_string()),
            style.font_size,
            &runs,
            None,
            None,
        )
        .ok()
        .and_then(|mut v| v.drain(..).next())
        .map(Arc::new)
}

/// Display text for an item's marker overlay. Unordered items
/// substitute a bullet glyph (`• `) when the cursor is outside, and
/// reveal the raw bullet character when the cursor is inside so the
/// user has visual feedback when "editing" the marker. Ordered
/// items always paint their digits — the numbers carry meaning and
/// users expect them stable.
fn list_marker_display_text(kind: ListItemKind, cursor_inside: bool) -> String {
    match kind {
        // Task items, cursor outside: render a single checkbox
        // glyph in place of the bullet+brackets. The user sees a
        // clean `☐ todo` / `☑ done` row.
        ListItemKind::Unordered(_, Some(true)) if !cursor_inside => "☑ ".to_string(),
        ListItemKind::Unordered(_, Some(false)) if !cursor_inside => "☐ ".to_string(),
        // Task items, cursor inside: render the raw source bytes
        // (`- [ ] ` / `- [x] `) so the user can navigate their
        // cursor into the brackets and toggle the checkbox by
        // typing `x` / space directly. The chrome paints in the
        // indent strip via the overlay-cursor path so the content
        // edge doesn't shift between focus states.
        ListItemKind::Unordered(b, Some(true)) => format!("{} [x] ", b as char),
        ListItemKind::Unordered(b, Some(false)) => format!("{} [ ] ", b as char),
        ListItemKind::Unordered(b, None) if cursor_inside => {
            let mut s = String::with_capacity(2);
            s.push(b as char);
            s.push(' ');
            s
        }
        ListItemKind::Unordered(_, None) => "• ".to_string(),
        ListItemKind::Ordered { number } => format!("{}. ", number),
    }
}

// ---------- Shaping ----------

struct ShapedLine {
    line: Arc<WrappedLine>,
    source_range: Range<usize>,
    display_to_source: Vec<usize>,
    is_delimiter: bool,
}

/// Total inner-content height contributed by a list of shaped lines:
/// the sum of per-line heights (each line's `line_height` * (wrap
/// boundaries + 1)), plus code-block breathing pads at every
/// fence↔content transition and after a trailing content tail. An
/// empty list reserves one `line_height` of space (the empty-block
/// fallback row that `prepaint` fabricates a shaped line for).
///
/// `request_layout` and `prepaint` both call this. Keeping the
/// arithmetic in one place is the only way to be certain the height
/// `request_layout` returned matches the height `prepaint` actually
/// fills with shaped lines.
#[allow(clippy::too_many_arguments)]
fn shaped_content_height(
    lines: &[ShapedLine],
    line_height: Pixels,
    inline_math: &[InlineMathPaintSpec],
    inline_images: &[InlineImagePaintSpec],
    body_ascent: Pixels,
    body_descent: Pixels,
    is_code: bool,
    style: &MarkdownStyle,
) -> Pixels {
    if lines.is_empty() {
        return line_height;
    }
    let mut h = px(0.0);
    for line in lines {
        // Math overshoot beyond body ascent + half-leading extends
        // the row above its natural top; analogous overshoot below
        // descent extends past the bottom. The row height passed to
        // gpui stays at `line_height` so the body baseline lands at
        // the same fraction of the row regardless of whether math
        // is present — surrounding lines visually align. Image
        // overshoot (image taller than line) extends both edges
        // symmetrically. Take the max of the two contributions per
        // edge so a row with both math and an image still reserves
        // enough space for the taller.
        let math_extra = compute_math_row_extra(
            &line.source_range,
            inline_math,
            body_ascent,
            body_descent,
            line_height,
        );
        let image_extra = compute_image_row_extra(&line.source_range, inline_images, line_height);
        let extra = combined_row_extra(math_extra, image_extra);
        let row_h = line_height + extra.total();
        h += row_h * ((line.line.wrap_boundaries().len() as f32) + 1.0);
    }
    if is_code {
        let mut last_was_delim = true; // "above the block" is a fence
        for line in lines {
            if line.is_delimiter != last_was_delim {
                h += style.code_block_content_padding_y;
            }
            last_was_delim = line.is_delimiter;
        }
        if !last_was_delim {
            h += style.code_block_content_padding_y;
        }
    }
    h
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
                is_delimiter: false,
            });
        }
        return out;
    }

    let block_is_code = matches!(block.kind, BlockKind::CodeBlock { .. });

    for raw_line in block_text.split_inclusive('\n') {
        let raw_end = cursor + raw_line.len();
        let trailing_nl = raw_line.ends_with('\n');
        let content_end = if trailing_nl { raw_end - 1 } else { raw_end };
        let logical_source_range = cursor..content_end;
        let line_source_range = cursor..raw_end;

        let (display_text, display_to_source) =
            build_display_line(source, &logical_source_range, block);

        // Code-block fence detection: the line's *full* logical
        // content is covered by either a hidden range (cursor
        // outside) or a dimmed inline run (cursor inside). For
        // headings the prefix `# ` is a delimiter range too, but it
        // doesn't cover the whole line — only fences do — so the
        // full-coverage check is a precise fence test inside code
        // blocks.
        let is_delimiter = block_is_code
            && logical_source_range.start < logical_source_range.end
            && line_is_fully_in_a_delimiter(&logical_source_range, block);

        // Hide-driven elision: if the line had visible source content
        // but every byte was hidden, drop the line — *unless* it's a
        // code-block delimiter, where we want the row to keep
        // reserving its space (so the block's height stays stable as
        // the cursor moves in/out and the fence rows can host a
        // distinct background).
        //
        // **Code-block bodies are also exempt.** An empty body line
        // inside a code block sits as a literal source line whose
        // bytes are typically just the chain prefix (BQ markers, LI
        // continuation indent). Those bytes are hidden by the chain
        // hide pass, so `display_text` ends up empty even though the
        // source line is meaningful (one empty row of code). Without
        // the exemption, the line gets dropped and the user sees
        // their empty rows collapse to nothing inside any
        // BQ/LI-wrapped code block.
        let was_empty_in_source = logical_source_range.start == logical_source_range.end;
        if display_text.is_empty() && !was_empty_in_source && !is_delimiter && !block_is_code {
            cursor = raw_end;
            if !trailing_nl {
                break;
            }
            continue;
        }

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
                is_delimiter,
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
                is_delimiter: false,
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

/// Does the shaped `line` fall on a delimiter (fence) row of the
/// block? The render layer lists those line ranges in
/// `block.delimiter_lines`; this checks whether the shaped line's
/// logical content range is covered by any of them. For code blocks
/// the fence rows are listed regardless of cursor position, so a
/// fence row with a partially-visible info string (` ```rust ` with
/// the cursor outside) still flags as a delimiter line for layout
/// purposes.
fn line_is_fully_in_a_delimiter(line: &Range<usize>, block: &RenderBlock) -> bool {
    block
        .delimiter_lines
        .iter()
        .any(|d| d.start <= line.start && d.end >= line.end)
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
        // Substitutions take precedence over hidden ranges and
        // raw-source copy: when `pos` matches a substitution's
        // source_range.start, append the substitution's display
        // bytes and skip past `source_range.end`. Each display byte
        // maps to `source_range.start` so a click on the substituted
        // glyph (e.g. a bullet) lands at the start of the original
        // marker bytes.
        if let Some(sub) = block
            .substitutions
            .iter()
            .find(|s| s.source_range.start == pos && s.source_range.end <= line.end)
        {
            for _ in 0..sub.display.len() {
                map.push(sub.source_range.start);
            }
            display.push_str(&sub.display);
            pos = sub.source_range.end;
            continue;
        }

        // Find the *furthest* hidden-range end that covers `pos`.
        // Multiple hidden ranges can overlap on the same byte — the
        // common case is a blockquote `> ` whose trailing space is
        // also the leading space of a code-block closing fence's
        // indent run. The previous `r.start == pos` predicate would
        // pick whichever range happened to start at `pos`, advance to
        // its end, then fail to skip the overlapping tail bytes of
        // the longer range. Finding the maximum `r.end` over all
        // ranges that cover `pos` skips the entire union in one step.
        //
        // The filter intentionally does *not* require `r.end <= line.end`:
        // a `BlockKind::DisplayMath` in display mode pushes one hide
        // covering the entire math source range (multiple lines), so
        // `r.end` extends past every individual shaped line's `line.end`.
        // We clamp `pos = end.min(line.end)` below so the advance never
        // walks past the current line — the cross-line span just means
        // every line this range covers contributes zero display bytes.
        let cover_end = block
            .hidden_ranges
            .iter()
            .filter(|r| r.start <= pos && pos < r.end)
            .map(|r| r.end)
            .max();
        if let Some(end) = cover_end {
            let mut jump = end.min(line.end);
            // A substitution whose start lies strictly inside the
            // covered span must still get its chance to apply. This
            // matters for table gap substitutions: the render layer's
            // `merge_hidden_ranges` merges the gap's chrome hide with
            // any *touching* construct-delimiter hide (a cell ending
            // in `` ` `` / `**` / `~~`), so the merged range starts
            // before the gap and a straight jump to its end would
            // skip the substitution entirely — the alignment pads
            // vanish exactly on styled cells. Clamp the jump to the
            // earliest in-span substitution start instead.
            if let Some(sub_start) = block
                .substitutions
                .iter()
                .filter(|s| s.source_range.start > pos && s.source_range.start < jump)
                .map(|s| s.source_range.start)
                .min()
            {
                jump = sub_start;
            }
            pos = jump;
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
        // Inline code swaps the run's font family to the inline-code
        // face (defaults to the mono family; see
        // `MarkdownStyle::inline_code_font_family` for why it's a
        // separate knob). Don't override the heading weight — bold
        // inline code inside a heading should still shape with the
        // heading's weight.
        if merged.code {
            run_font.family = style.inline_code_font_family.clone();
        }
        if merged.bold || base_weight == FontWeight::BOLD {
            run_font.weight = FontWeight::BOLD;
        } else if merged.table_header {
            // Table header cells: the table-header weight (Medium by
            // default) carries the emphasis; explicit bold above wins.
            run_font.weight = style.table_header_weight;
        } else if base_weight != FontWeight::NORMAL {
            run_font.weight = base_weight;
        }
        if merged.italic {
            run_font.style = FontStyle::Italic;
        }

        let color = if merged.dimmed {
            style.delimiter_color
        } else if merged.link {
            style.link_color
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

        // Inline code: paint a faint background under the run so the
        // span reads as a chip. Dim takes precedence (the delimiter
        // backticks share the run color but should *not* paint a
        // background — only the content does).
        let background_color = if merged.code && !merged.dimmed {
            Some(style.inline_code_background)
        } else {
            None
        };

        // Inline link: single underline beneath the link text. No
        // underline on the bracket / url delimiters (those are
        // hidden when the cursor is outside, dimmed when inside).
        let underline = if merged.link && !merged.dimmed {
            Some(gpui::UnderlineStyle {
                thickness: px(1.0),
                color: Some(color),
                wavy: false,
            })
        } else {
            None
        };

        runs.push(TextRun {
            len: j - i,
            font: run_font,
            color,
            background_color,
            underline,
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

fn base_font_for_block(kind: &BlockKind, style: &MarkdownStyle) -> gpui::Font {
    let family = match kind {
        BlockKind::CodeBlock { .. } => style.mono_font_family.clone(),
        _ => style.font_family.clone(),
    };
    gpui::Font {
        family,
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
        Some(l) => style.weight_for_heading(l),
        None => FontWeight::NORMAL,
    }
}

// ---------- Cursor / selection ----------

/// A paint quad tagged by which kind of line it belongs to. Code
/// blocks paint delimiter quads outside the content mask (so fence-row
/// cursors / selections aren't clipped) and content quads inside the
/// mask (so they share the content area's clipping rectangle). For
/// non-code blocks every quad is `is_delimiter == false` and both
/// branches paint the same way.
#[derive(Clone)]
struct TaggedQuad {
    quad: gpui::PaintQuad,
    is_delimiter: bool,
}

#[allow(clippy::too_many_arguments)]
fn build_caret_and_selection(
    block: &LaidOutBlock,
    selection: Selection,
    style: &MarkdownStyle,
    is_last_block: bool,
    next_block_start: Option<usize>,
    marker_overlay_caret: Option<TaggedQuad>,
    caret_downstream: bool,
) -> (Option<TaggedQuad>, Vec<TaggedQuad>) {
    let cursor_offset = selection.head();
    let cursor_color = style.caret_color;
    let selection_color = style.selection_color;

    let mut cursor: Option<TaggedQuad> = marker_overlay_caret;
    let mut boundary_fallback: Option<TaggedQuad> = None;
    let mut sel_quads: Vec<TaggedQuad> = Vec::new();

    let sel_range = selection.selection_range();
    let has_selection = sel_range.start != sel_range.end;

    let cursor_in_block = block_claims_cursor(
        cursor_offset,
        block.source_range.start,
        block.source_range.end,
        next_block_start,
    );

    for line in &block.lines {
        let lo = line.source_range.start;
        let hi = line.source_range.end;
        if cursor_in_block && cursor.is_none() {
            let strict = cursor_offset >= lo && cursor_offset < hi;
            let boundary = cursor_offset == hi && (is_last_block || hi == block.source_range.end);
            if strict || (boundary && boundary_fallback.is_none()) {
                let local =
                    line.local_position_for_source_offset_biased(cursor_offset, caret_downstream);
                let x = line.origin.x + local.x;
                let y = line.origin.y + local.y;
                let quad = TaggedQuad {
                    quad: fill(
                        Bounds::new(point(x, y), size(px(2.0), line.row_height)),
                        cursor_color,
                    ),
                    is_delimiter: line.is_delimiter,
                };
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

/// Wash quads for one highlight layer's ranges over one block: for each
/// (pre-merged) range that intersects a laid-out line, the same quad geometry
/// as a selection (via [`paint_selection_for_line`]) in `color` — the layer's
/// wash (`MarkdownStyle::highlight_layer_color`). Inert with respect to the
/// caret and selection — pure decoration under the text.
fn build_highlight_quads(
    block: &LaidOutBlock,
    ranges: &[std::ops::Range<usize>],
    color: gpui::Hsla,
) -> Vec<TaggedQuad> {
    if ranges.is_empty() {
        return Vec::new();
    }
    let mut quads = Vec::new();
    for line in &block.lines {
        let lo = line.source_range.start;
        let hi = line.source_range.end;
        for range in ranges {
            let lo_clamped = range.start.max(lo);
            let hi_clamped = range.end.min(hi);
            if hi_clamped > lo_clamped {
                paint_selection_for_line(line, lo_clamped, hi_clamped, hi, color, &mut quads);
            }
        }
    }
    quads
}

fn paint_selection_for_line(
    line: &LaidOutLine,
    lo: usize,
    hi: usize,
    line_hi: usize,
    color: gpui::Hsla,
    out: &mut Vec<TaggedQuad>,
) {
    let push = |q: gpui::PaintQuad, out: &mut Vec<TaggedQuad>| {
        out.push(TaggedQuad {
            quad: q,
            is_delimiter: line.is_delimiter,
        });
    };
    let start = line.local_position_for_source_offset(lo);
    let end = line.local_position_for_source_offset(hi);
    let row_height = line.row_height;
    let eol_pad = if hi == line_hi { px(6.0) } else { px(0.0) };

    if start.y == end.y {
        let x0 = line.origin.x + start.x;
        let x1 = line.origin.x + end.x + eol_pad;
        let y0 = line.origin.y + start.y;
        push(
            fill(
                Bounds::from_corners(point(x0, y0), point(x1, y0 + row_height)),
                color,
            ),
            out,
        );
        return;
    }

    let row_count = line.row_count();
    let line_width = line.line.width();
    let start_row = (f32::from(start.y) / f32::from(row_height)).round() as usize;
    let end_row = (f32::from(end.y) / f32::from(row_height)).round() as usize;

    let y_start = line.origin.y + start.y;
    push(
        fill(
            Bounds::from_corners(
                point(line.origin.x + start.x, y_start),
                point(line.origin.x + line_width, y_start + row_height),
            ),
            color,
        ),
        out,
    );

    for row in (start_row + 1)..end_row.min(row_count) {
        let y = line.origin.y + row_height * (row as f32);
        push(
            fill(
                Bounds::from_corners(
                    point(line.origin.x, y),
                    point(line.origin.x + line_width, y + row_height),
                ),
                color,
            ),
            out,
        );
    }

    let y_end = line.origin.y + end.y;
    push(
        fill(
            Bounds::from_corners(
                point(line.origin.x, y_end),
                point(line.origin.x + end.x + eol_pad, y_end + row_height),
            ),
            color,
        ),
        out,
    );
}

/// Decide whether a block claims the cursor at `cursor_offset`. A block
/// owns the offset if it sits strictly inside the block's source range,
/// OR if it sits at the block's end *and* no following block starts at
/// that offset. The end-clause keeps the trailing cursor at end-of-doc
/// rendered (no next block to claim it) while preventing double-paint
/// when a trailing or leading synthetic empty starts at the previous
/// block's end — `inject_empty_paragraphs` does exactly that for those
/// cases, so without this guard a cursor at the boundary would paint
/// on both blocks' rows.
fn block_claims_cursor(
    cursor_offset: usize,
    block_start: usize,
    block_end: usize,
    next_block_start: Option<usize>,
) -> bool {
    let strict = cursor_offset >= block_start && cursor_offset < block_end;
    let at_end = cursor_offset == block_end;
    let next_claims = matches!(next_block_start, Some(s) if s == cursor_offset);
    strict || (at_end && !next_claims)
}

/// Strip the chain continuation prefix (`> `, list-item indent, …) from
/// every line of `latex` that carries it. Source bytes for block-level
/// `$$..$$` math that lives inside a blockquote or list item include the
/// scope-continuation prefix on every line past the opener (pulldown
/// preserves the raw source slice; only its emitted Cow content is
/// stripped); RaTeX would choke on a stray `> ` mid-equation, so we
/// strip those bytes here before handing the LaTeX to `math::typeset`.
///
/// Returns `Cow::Borrowed(latex)` when the chain is empty or no line in
/// fact starts with the prefix — no allocation in the common case
/// (top-level math, or single-line math whose raw bytes never needed
/// prefix continuation).
/// Typeset a block-level `$$..$$` math construct in display mode,
/// applying [`sanitize_latex`] first. Returns `None` when the content
/// range is out of bounds, when the block isn't a `BlockKind::DisplayMath`,
/// or when RaTeX rejects the LaTeX — the caller falls through to the
/// regular text-shape path (which renders the raw source as a fallback).
///
/// Consolidates the sanitize + typeset pattern used in `request_layout`
/// (display-mode natural height and edit-mode min-height), `prepaint`
/// (display-mode `MathPaint`), and any future caller. Each phase still
/// re-typesets — we don't memoize across phases yet — but every site
/// goes through the same code path, so a future move to a memoized
/// version only needs to change one function.
fn typeset_display_math_block(
    block: &crate::render_spec::RenderBlock,
    source: &str,
) -> Option<crate::math::MathLayout> {
    let content_range = match &block.kind {
        BlockKind::DisplayMath { content_range, .. } => content_range.clone(),
        _ => return None,
    };
    let latex = source.get(content_range)?;
    let sanitized = sanitize_latex(latex, &block.containers);
    crate::math::typeset(&sanitized, crate::math::MathMode::Display).ok()
}

fn sanitize_latex<'a>(latex: &'a str, containers: &[Container]) -> std::borrow::Cow<'a, str> {
    if containers.is_empty() {
        return std::borrow::Cow::Borrowed(latex);
    }
    let prefix = crate::render_spec::containers_continuation_prefix(containers);
    if prefix.is_empty() {
        return std::borrow::Cow::Borrowed(latex);
    }

    let has_prefix = latex.split('\n').any(|line| line.starts_with(&prefix));
    if !has_prefix {
        return std::borrow::Cow::Borrowed(latex);
    }

    let mut sanitized = String::with_capacity(latex.len());
    for (i, line) in latex.split('\n').enumerate() {
        if i > 0 {
            sanitized.push('\n');
        }
        if line.starts_with(&prefix) {
            sanitized.push_str(&line[prefix.len()..]);
        } else {
            sanitized.push_str(line);
        }
    }
    std::borrow::Cow::Owned(sanitized)
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
            ..Default::default()
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

    /// Shape each block and return the cursor's local-x position
    /// for the supplied source offset (relative to the block's
    /// origin). Used to verify the cursor lands at a non-zero
    /// column for end-of-line cases that include trailing
    /// whitespace.
    fn cursor_x_for_offset(cx: &mut TestAppContext, src: &str, offset: usize) -> Pixels {
        let state = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(offset),
            ..Default::default()
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
            for b in &blocks {
                let font_size = font_size_for_block(&b.kind, &style);
                let line_height = font_size * style.line_height.0;
                let shaped =
                    shape_block_lines(&src_owned, b, &style, font_size, Some(px(720.0)), window);
                for sl in shaped {
                    if sl.source_range.start <= offset && offset <= sl.source_range.end {
                        let laid = LaidOutLine {
                            line: sl.line,
                            origin: point(px(0.0), px(0.0)),
                            row_height: line_height,
                            wrapped_height: line_height,
                            source_range: sl.source_range,
                            display_to_source: sl.display_to_source,
                            is_delimiter: sl.is_delimiter,
                        };
                        return laid.local_position_for_source_offset(offset).x;
                    }
                }
            }
            px(0.0)
        })
        .expect("update window")
    }

    #[gpui::test]
    fn cursor_at_end_of_list_item_with_trailing_space_lands_past_content(cx: &mut TestAppContext) {
        // `- foo ` (trailing space, no newline). Cursor at byte 6.
        // The display line is `foo ` (`- ` hidden as marker chrome).
        // The cursor must land at a non-zero column — i.e. *past*
        // the `foo` content — even though pulldown's parsed leaf
        // range typically excludes trailing whitespace. Without
        // extending the block's source_range to swallow trailing
        // whitespace on the last content line, the byte at the
        // cursor's position falls outside every block and no caret
        // paints at all.
        let plain_x = cursor_x_for_offset(cx, "- foo", 5);
        let trailing_x = cursor_x_for_offset(cx, "- foo ", 6);
        assert!(
            trailing_x > plain_x,
            "trailing-space cursor x ({:?}) should be greater than no-trail ({:?})",
            trailing_x,
            plain_x,
        );
    }

    /// Shape a table document's first row through the real table
    /// augment pass (edit mode, caret at `cursor`) and return the
    /// local x of each probe offset plus the shaped line's width.
    fn table_line_x_positions(
        cx: &mut TestAppContext,
        src: &str,
        cursor: usize,
        probes: &[usize],
    ) -> (Vec<Pixels>, Pixels) {
        let state = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(cursor),
            ..Default::default()
        };
        let tree = parse(src);
        let blocks = render(&state, &tree).blocks;
        let src_owned = src.to_string();
        let probes = probes.to_vec();

        let handle = cx.update(|cx| {
            gpui_component::init(cx);
            cx.open_window(WindowOptions::default(), |_, cx| cx.new(|_| EmptyRoot))
                .expect("open window")
        });

        cx.update_window(handle.into(), |_, window, cx| {
            let style = MarkdownStyle::from_theme(cx);
            let block = blocks
                .iter()
                .find(|b| matches!(b.kind, BlockKind::Table { .. }))
                .expect("a table block");
            let font_size = font_size_for_block(&block.kind, &style);
            let line_height = font_size * style.line_height.0;
            let layout = layout_table(block, &src_owned, font_size, &style, px(720.0), window)
                .expect("a table layout");
            let mut out = Vec::new();
            for probe in &probes {
                let piece = layout
                    .pieces
                    .iter()
                    .find(|p| p.source_range.start <= *probe && *probe <= p.source_range.end)
                    .expect("a table piece containing the probe");
                let laid = LaidOutLine {
                    line: piece.line.clone(),
                    origin: point(piece.rel_origin.x, piece.rel_origin.y),
                    row_height: line_height,
                    wrapped_height: piece.wrapped_height,
                    source_range: piece.source_range.clone(),
                    display_to_source: piece.display_to_source.clone(),
                    is_delimiter: false,
                };
                let local = laid.local_position_for_source_offset(*probe);
                out.push(laid.origin.x + local.x);
            }
            (out, layout.size.width)
        })
        .expect("update window")
    }

    #[gpui::test]
    fn table_caret_after_trailing_space_renders_inside_its_cell(cx: &mut TestAppContext) {
        // Type a trailing space at the end of a mid-row cell:
        // `| ab  | cd |` with the caret after the space (offset 5).
        // The caret must render inside the first cell's box — left of
        // the second cell's content — and the second cell's x must
        // not shift versus the canonical `| ab | cd |` (the space
        // absorbs into the boundary fill). This was Mike's
        // trailing-space bug: the whole-chrome substitution swallowed
        // the typed space, painting the caret past the *next* pipe
        // and shifting the rest of the row right.
        let spaced = "| ab  | cd |\n| --- | --- |\n| x | y |\n";
        let caret = 5; // after the typed trailing space
        let cd = spaced.find("cd").unwrap();
        let y = spaced.find("| y |").unwrap() + 2;
        let (xs, _) = table_line_x_positions(cx, spaced, caret, &[caret, cd, y]);
        let (caret_x, cd_x, y_x) = (xs[0], xs[1], xs[2]);
        assert!(
            caret_x < cd_x,
            "caret after a trailing space must stay left of the next cell \
             (caret {caret_x:?}, next cell {cd_x:?})",
        );
        // The typed space counts as occupied width, so the whole
        // column widens *together*: the second column stays aligned
        // across rows (within pad-glyph rounding) instead of only the
        // caret's row shifting right.
        let skew = (cd_x - y_x).abs();
        assert!(
            skew < px(4.0),
            "the second column must stay aligned across rows while a \
             trailing space is being typed (cd at {cd_x:?}, y at {y_x:?})",
        );
    }

    #[gpui::test]
    fn table_caret_after_trailing_space_in_last_cell_stays_inside_the_row(cx: &mut TestAppContext) {
        // Same bug at the row's last cell: `| ab | cd  |` caret after
        // the typed space (before the trailing pipe). The caret must
        // render left of the shaped row's right edge (it used to
        // paint past the trailing pipe, outside the table).
        let spaced = "| ab | cd  |\n| --- | --- |\n| x | y |\n";
        let caret = spaced.find("cd").unwrap() + 3; // after the typed space
        let (xs, line_width) = table_line_x_positions(cx, spaced, caret, &[caret]);
        assert!(
            xs[0] < line_width,
            "caret after a trailing space in the last cell must stay inside \
             the row (caret {:?}, row width {line_width:?})",
            xs[0],
        );
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
    fn empty_body_line_in_top_level_code_block_shapes_one_row(cx: &mut TestAppContext) {
        // ```js\n\n``` — opener, one empty body line, closer. Shape
        // count: 3 lines (opener + body + closer).
        let counts = shape_visible_row_counts(cx, "```js\n\n```");
        assert_eq!(counts, vec![3]);
    }

    #[gpui::test]
    fn empty_body_line_in_bq_wrapped_code_block_shapes_one_row(cx: &mut TestAppContext) {
        // BQ-wrapped fence with one empty body line. Each line in the
        // body carries a `> ` prefix that the chain hide pass marks
        // hidden — without the code-block exemption in
        // `shape_block_lines`, the all-hidden body line gets dropped
        // and the empty row visually disappears (the user-reported
        // bug). With the exemption, the body line shapes as one
        // visible empty row.
        let counts = shape_visible_row_counts(cx, "> ```js\n> \n> ```\n");
        // One block (the code block); 3 visible rows (opener, body, closer).
        assert_eq!(counts, vec![3]);
    }

    #[gpui::test]
    fn empty_body_line_in_li_wrapped_code_block_shapes_one_row(cx: &mut TestAppContext) {
        // Same shape with an LI wrapper. The body line is `   ` (3
        // spaces of LI continuation indent, all hidden by the chain
        // hide pass). Without the exemption, dropped → empty row
        // disappears. With it, shapes as one visible empty row.
        let counts = shape_visible_row_counts(cx, "1. ```js\n   \n   ```\n");
        assert_eq!(counts, vec![3]);
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

    // ---- Cursor-claim rule: each offset is owned by at most one block ---

    /// Render `src` and walk every byte position from 0 to `len`, asking
    /// each block whether it claims that offset. Returns
    /// `claims_by_offset[p]` = list of block indices that claim `p`. The
    /// invariant under test: every entry has length exactly 1 — no
    /// double-paint, no orphan offset.
    fn cursor_claims_per_offset(src: &str) -> Vec<Vec<usize>> {
        let state = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(0),
            ..Default::default()
        };
        let tree = parse(src);
        let blocks = render(&state, &tree).blocks;
        let starts: Vec<usize> = blocks.iter().map(|b| b.source_range.start).collect();
        (0..=src.len())
            .map(|offset| {
                blocks
                    .iter()
                    .enumerate()
                    .filter(|(idx, b)| {
                        let next_start = starts.get(idx + 1).copied();
                        block_claims_cursor(
                            offset,
                            b.source_range.start,
                            b.source_range.end,
                            next_start,
                        )
                    })
                    .map(|(idx, _)| idx)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn no_offset_is_claimed_by_more_than_one_block() {
        // The no-double-cursor invariant. Forbidden offsets may go
        // unclaimed (zero blocks) — fine, the snap rule keeps the
        // cursor away from them. Allowed offsets must be claimed by
        // exactly one block. The bug we're regressing against had two
        // blocks claiming the same boundary offset (e.g., end of
        // "paragraph" AND start of trailing empty), painting two
        // carets at the same offset.
        for src in [
            "p1\n\np2",
            "p1\n\n\n\np2",
            "p1\n\n\n\n\n\np2",
            "paragraph",
            "paragraph\n\n",
            "paragraph\n\n\n\n",
            "paragraph\n\n\n\n\n\n",
            "\n\np1",
            "\n\n\n\np1",
            "# title\n\nbody",
            "# title\n\n\n\n\n\nbody",
        ] {
            for (offset, claims) in cursor_claims_per_offset(src).into_iter().enumerate() {
                assert!(
                    claims.len() <= 1,
                    "offset {offset} in {src:?} claimed by multiple blocks {claims:?}"
                );
                if !crate::analysis::is_forbidden_position(src, offset) {
                    assert_eq!(
                        claims.len(),
                        1,
                        "allowed offset {offset} in {src:?} claimed by no block"
                    );
                }
            }
        }
    }

    #[test]
    fn trailing_empty_layout_keeps_paragraph_end_on_paragraph_row() {
        // In `paragraph\n\n\n\n\n\n` (paragraph + 3 trailing empties),
        // the trailing-pair formula offsets each empty by 1 inside the
        // gap so its strict-interior position falls between two pairs
        // — that's where the cursor naturally rests when it's on an
        // empty row, and typing there creates a new paragraph for the
        // row instead of extending the previous content paragraph.
        //
        // Layout: paragraph (0..9), empty 1 (10..12), empty 2 (12..14),
        // empty 3 (14..15) — last empty clamped to doc end. The
        // resting positions are 11, 13, 15 (boundaries between pairs
        // and end-of-doc); position 9 stays on paragraph.
        let claims = cursor_claims_per_offset("paragraph\n\n\n\n\n\n");
        assert_eq!(claims[9], vec![0]); // end of paragraph
        assert_eq!(claims[11], vec![1]); // empty 1 strict interior
        assert_eq!(claims[13], vec![2]); // empty 2 strict interior
        assert_eq!(claims[15], vec![3]); // empty 3 (end-of-doc, last block)
    }

    #[test]
    fn leading_empty_boundary_claimed_by_paragraph_not_empty() {
        // Symmetric: in `\n\np1`, byte 2 is end of leading empty AND
        // start of "p1". p1 strictly contains it; empty yields.
        let claims = cursor_claims_per_offset("\n\np1");
        // Block 0: empty (0..2). Block 1: p1 (2..4).
        assert_eq!(claims[0], vec![0]);
        assert_eq!(claims[2], vec![1]);
    }

    #[test]
    fn end_of_doc_still_renders_on_last_block() {
        // No next block to claim end-of-doc; the last block must keep
        // its end-clause claim.
        let claims = cursor_claims_per_offset("paragraph");
        assert_eq!(claims.last().unwrap(), &vec![0]);
    }

    #[test]
    fn empty_doc_offset_zero_claimed_by_anchor_block() {
        let claims = cursor_claims_per_offset("");
        assert_eq!(claims, vec![vec![0]]);
    }

    /// Run `augment_block_with_math` against the first paragraph
    /// block of `src` and return per-line row-extra totals (extra
    /// vertical space the math overlay forces *beyond* the body
    /// line height).
    fn math_row_extra_totals(cx: &mut TestAppContext, src: &str) -> Vec<Pixels> {
        let state = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(0),
            ..Default::default()
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
            let block = blocks
                .iter()
                .find(|b| matches!(b.kind, BlockKind::Paragraph))
                .expect("paragraph block");
            let font_size = font_size_for_block(&block.kind, &style);
            let line_height = font_size * style.line_height.0;
            let (_aug, paint_specs) =
                augment_block_with_math(block, &src_owned, font_size, &style, window);
            let (body_ascent, body_descent) =
                body_metrics_for_block(&block.kind, &style, font_size, window);
            let shaped = shape_block_lines(
                &src_owned,
                block,
                &style,
                font_size,
                Some(px(720.0)),
                window,
            );
            shaped
                .into_iter()
                .map(|sl| {
                    compute_math_row_extra(
                        &sl.source_range,
                        &paint_specs,
                        body_ascent,
                        body_descent,
                        line_height,
                    )
                    .total()
                })
                .collect()
        })
        .expect("update window")
    }

    #[gpui::test]
    fn plain_paragraph_has_zero_math_row_extra(cx: &mut TestAppContext) {
        // Math-free lines never extend their row reservation —
        // the body font's natural ascent / descent already covers
        // the shaped text.
        let totals = math_row_extra_totals(cx, "just plain prose");
        assert!(totals.iter().all(|p| *p == px(0.0)));
    }

    #[gpui::test]
    fn tall_inline_math_extends_row_reservation(cx: &mut TestAppContext) {
        // `$\frac{a}{b}$` typesets a fraction whose total height
        // (numerator + bar + denominator) is taller than the body
        // font's ascent + descent + leading at the same em size.
        // The row extra must be > 0 — that's the layout-level
        // guarantee the math doesn't clip into adjacent rows.
        let totals = math_row_extra_totals(cx, r"see $\frac{a}{b}$ here");
        assert_eq!(totals.len(), 1, "single shaped line");
        assert!(
            totals[0] > px(0.0),
            "tall fraction must extend row reservation (got {:?})",
            totals[0]
        );
    }

    #[gpui::test]
    fn tall_math_extends_row_more_than_short_math(cx: &mut TestAppContext) {
        // The extra scales with the math's ink overshoot beyond
        // the body font's half-leading. A single italic letter
        // (`$x$`) typically fits inside the half-leading and adds
        // zero extra; a fraction (`$\frac{a}{b}$`) overshoots
        // significantly. We don't pin an exact px value because
        // the body-vs-math metric delta is font-bound; the
        // layout-level invariant is just that tall math extends
        // strictly more than short math.
        let short = math_row_extra_totals(cx, "see $x$ here");
        let tall = math_row_extra_totals(cx, r"see $\frac{a}{b}$ here");
        assert_eq!(short.len(), 1);
        assert_eq!(tall.len(), 1);
        assert!(
            tall[0] > short[0] + px(2.0),
            "tall math should add noticeably more row extra than short math: tall={:?} short={:?}",
            tall[0],
            short[0],
        );
    }
}
