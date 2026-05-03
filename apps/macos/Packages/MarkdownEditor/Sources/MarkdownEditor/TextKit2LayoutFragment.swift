import AppKit

/// Custom `NSTextLayoutFragment` that paints blockquote left borders and
/// widens its rendering surface for paragraphs that host a
/// `BlockAttachment` view. Code-block backgrounds (the original Phase 3
/// motivation for this subclass) are no longer painted here — Phase 2.2
/// retired the per-fragment painting path in favour of the embedded
/// `CodeBlockRenderer`, which paints its own background inside the
/// attachment view.
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
/// and the blockquote borders / attachment views get stale or clipped
/// pixels on resize / scroll. We widen the surface bounds whenever any
/// decoration is present.
@MainActor
final class TextKit2LayoutFragment: NSTextLayoutFragment {

  // NSTextLayoutFragment's draw(at:in:) and renderingSurfaceBounds overrides
  // are inherited as nonisolated, so we mark these stored properties
  // `nonisolated(unsafe)` and only mutate them on the main actor (vend-time
  // and resize).

  /// One x-position per blockquote nesting level that contains this
  /// paragraph. Empty when not inside any blockquote.
  nonisolated(unsafe) var blockquoteBorderXPositions: [CGFloat] = []

  /// Width of the enclosing text container in container coordinates. Used
  /// to widen the rendering surface for full-container decorations.
  /// Updated at vend time and on container resize.
  nonisolated(unsafe) var containerWidth: CGFloat = 0

  /// Phase 2 bridging-layer: set by the layout-manager delegate when this
  /// fragment vends a paragraph that contains a `BlockAttachment`. When
  /// `true`, `renderingSurfaceBounds` widens to encompass the attachment's
  /// reserved region (which can be substantially taller than the host
  /// glyph line) so AppKit doesn't clip the embedded view's redraw.
  nonisolated(unsafe) var containsBlockAttachment: Bool = false

  /// Reserved height of the attachment, in points. Carried alongside
  /// `containsBlockAttachment` so the surface-bounds widening can size
  /// itself even before the attachment view has reported its frame.
  nonisolated(unsafe) var blockAttachmentReservedHeight: CGFloat = 0

  // MARK: - Style

  nonisolated(unsafe) var blockquoteBorderColor: NSColor = .separatorColor
  nonisolated(unsafe) var blockquoteBorderWidth: CGFloat = 3

  // MARK: - Drawing

  override func draw(at point: CGPoint, in context: CGContext) {
    let hasBlockquote = !blockquoteBorderXPositions.isEmpty

    if hasBlockquote {
      // The fragment frame is in container coords; the CGContext has been
      // pre-translated so local (0, 0) maps to `point` in container space.
      // To convert a container-coords x into a local x, subtract the
      // fragment's container-coords origin.x.
      let frame = layoutFragmentFrame
      let localOriginX = -frame.origin.x

      // Pixel-snap the draw rect's Y bounds to integer container-coord
      // values with `round` for both edges so adjacent paragraphs tile
      // flush (no sub-pixel hairlines) under the blockquote border.
      let globalTop = round(frame.origin.y)
      let globalBottom = round(frame.origin.y + frame.height)
      let snappedLocalY = globalTop - frame.origin.y
      let snappedHeight = max(0, globalBottom - globalTop)

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

    // Glyphs draw last so text always sits on top of decorations.
    super.draw(at: point, in: context)
  }

  override var renderingSurfaceBounds: CGRect {
    let glyphBounds = super.renderingSurfaceBounds
    guard !blockquoteBorderXPositions.isEmpty
      || containsBlockAttachment
    else {
      return glyphBounds
    }

    // Widen the dirty region so AppKit doesn't clip our left borders or
    // attachment views. The surface bounds are in fragment-local
    // coordinates. Match the pixel-snapped Y bounds used by
    // `draw(at:in:)` so AppKit doesn't clip rows at the snapped top /
    // bottom edge.
    let frame = layoutFragmentFrame
    let localOriginX = -frame.origin.x
    let snappedLocalY = round(frame.origin.y) - frame.origin.y
    let snappedHeight = max(0, round(frame.origin.y + frame.height) - round(frame.origin.y))
    // When the fragment hosts a block attachment, the attachment's view
    // can be much taller than the host glyph line. Widen the height to
    // the spec's reserved region so AppKit's clip doesn't truncate the
    // embedded view's redraw region.
    let attachmentHeight: CGFloat = containsBlockAttachment ? blockAttachmentReservedHeight : 0
    let widened = CGRect(
      x: localOriginX,
      y: min(glyphBounds.minY, snappedLocalY),
      width: max(containerWidth, glyphBounds.maxX - localOriginX),
      height: max(snappedHeight, glyphBounds.height, attachmentHeight))
    return widened.union(glyphBounds)
  }
}
