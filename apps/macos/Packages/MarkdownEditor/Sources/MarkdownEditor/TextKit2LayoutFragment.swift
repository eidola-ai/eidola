import AppKit

/// Phase 3 of the TextKit 2 migration: a custom `NSTextLayoutFragment` that
/// paints code-block backgrounds and blockquote left borders, replacing the
/// TK1-only `CodeBlockBackgroundLayoutManager`.
///
/// One paragraph can be both inside a code block AND inside one or more
/// blockquotes (e.g. a code block nested in a blockquote), so a single
/// fragment subclass handles both decorations rather than splitting into two.
///
/// ## Coordinate spaces (per Spike 2 findings)
///
/// - `layoutFragmentFrame` is in **container coordinates**.
/// - `renderingSurfaceBounds` is in **fragment-local coordinates** (relative
///   to the `point` argument of `draw(at:in:)`).
/// - Inside `draw(at:in:)`, the CGContext is pre-translated so `(0, 0)` maps
///   to `point` in container space — to fill from container `x = 0` we draw
///   at local `x = -layoutFragmentFrame.origin.x`.
///
/// ## Why we override `renderingSurfaceBounds`
///
/// Without an override, AppKit clips the dirty region to the glyph extent
/// and full-width backgrounds get stale/clipped pixels on resize/scroll.
/// We widen the surface bounds to cover the full container width whenever
/// any decoration is present.
@MainActor
final class TextKit2LayoutFragment: NSTextLayoutFragment {

  // NSTextLayoutFragment's draw(at:in:) and renderingSurfaceBounds overrides
  // are inherited as nonisolated, so we mark these stored properties
  // `nonisolated(unsafe)` and only mutate them on the main actor (vend-time
  // and resize). Same pattern as CodeBlockBackgroundLayoutManager.

  /// Set by the layout-manager delegate at vend time. `nil` means this
  /// paragraph is not inside a code block.
  nonisolated(unsafe) var codeBlockOrigin: CGFloat?

  /// One x-position per blockquote nesting level that contains this
  /// paragraph. Empty when not inside any blockquote.
  nonisolated(unsafe) var blockquoteBorderXPositions: [CGFloat] = []

  /// Width of the enclosing text container in container coordinates. Used to
  /// compute the right edge of the full-width code-block background. Updated
  /// at vend time and on container resize.
  nonisolated(unsafe) var containerWidth: CGFloat = 0

  // MARK: - Style

  /// Mirrors `CodeBlockBackgroundLayoutManager`'s default — translucent so
  /// the system selection highlight (drawn underneath by NSTextLayoutManager)
  /// remains clearly visible.
  nonisolated(unsafe) var codeBlockBackgroundColor: NSColor =
    .quaternaryLabelColor.withAlphaComponent(0.5)

  nonisolated(unsafe) var blockquoteBorderColor: NSColor = .separatorColor
  nonisolated(unsafe) var blockquoteBorderWidth: CGFloat = 3

  // MARK: - Drawing

  override func draw(at point: CGPoint, in context: CGContext) {
    let hasCodeBg = codeBlockOrigin != nil
    let hasBlockquote = !blockquoteBorderXPositions.isEmpty

    if hasCodeBg || hasBlockquote {
      // The fragment frame is in container coords; the CGContext has been
      // pre-translated so local (0, 0) maps to `point` in container space.
      // To convert a container-coords x into a local x, subtract the
      // fragment's container-coords origin.x.
      let frame = layoutFragmentFrame
      let localOriginX = -frame.origin.x

      // A code block is split by TK2 into one fragment per source paragraph
      // (one per line of code), and each fragment paints its own background
      // rectangle. With non-integer fragment Y values plus screen-pixel
      // rasterization, a naive `[0, frame.height]` rect leaves thin gaps
      // (CG fill of fractional rects partially covers the bottom row) or
      // overlaps by a sub-pixel row at every line boundary — visible as
      // hairlines breaking up what should read as one continuous block.
      //
      // Fix: pixel-snap the draw rect's Y bounds to integer container-coord
      // values with `round` for both edges. With the renderer's content-
      // paragraph `paragraphSpacing = 0`, the next fragment's `origin.y`
      // equals this fragment's `origin.y + height` exactly, so rounding
      // both edges with the same rule produces a shared integer boundary
      // — adjacent fragments tile flush. CG `fill` uses half-open Y
      // semantics (paints `[top, top + height)`), so the boundary row
      // belongs to exactly one fragment: no gap, no overlap.
      let globalTop = round(frame.origin.y)
      let globalBottom = round(frame.origin.y + frame.height)
      let snappedLocalY = globalTop - frame.origin.y
      let snappedHeight = max(0, globalBottom - globalTop)

      // Fragment-local rectangle that covers the entire wrapped paragraph
      // height. Per Spike 2, `layoutFragmentFrame.height` already covers all
      // wrapped lines — no need to enumerate `textLineFragments`.
      if hasCodeBg, let xOrigin = codeBlockOrigin {
        let bgRect = CGRect(
          x: localOriginX + xOrigin,
          y: snappedLocalY,
          width: max(0, containerWidth - xOrigin),
          height: snappedHeight)
        context.saveGState()
        context.setFillColor(codeBlockBackgroundColor.cgColor)
        context.fill(bgRect)
        context.restoreGState()
      }

      if hasBlockquote {
        context.saveGState()
        context.setFillColor(blockquoteBorderColor.cgColor)
        for xPosition in blockquoteBorderXPositions {
          let borderRect = CGRect(
            x: localOriginX + xPosition,
            y: snappedLocalY,
            width: blockquoteBorderWidth,
            height: snappedHeight)
          context.fill(borderRect)
        }
        context.restoreGState()
      }
    }

    // Glyphs draw last so text always sits on top of decorations.
    super.draw(at: point, in: context)
  }

  override var renderingSurfaceBounds: CGRect {
    let glyphBounds = super.renderingSurfaceBounds
    guard codeBlockOrigin != nil || !blockquoteBorderXPositions.isEmpty else {
      return glyphBounds
    }

    // Widen the dirty region so AppKit doesn't clip our full-width fill /
    // left borders. The surface bounds are in fragment-local coordinates.
    // Match the pixel-snapped Y bounds used by `draw(at:in:)` so AppKit
    // doesn't clip rows at the snapped top/bottom edge.
    let frame = layoutFragmentFrame
    let localOriginX = -frame.origin.x
    let snappedLocalY = round(frame.origin.y) - frame.origin.y
    let snappedHeight = max(0, round(frame.origin.y + frame.height) - round(frame.origin.y))
    let widened = CGRect(
      x: localOriginX,
      y: min(glyphBounds.minY, snappedLocalY),
      width: max(containerWidth, glyphBounds.maxX - localOriginX),
      height: max(snappedHeight, glyphBounds.height))
    return widened.union(glyphBounds)
  }
}
