import AppKit

/// Phase 3 of the TextKit 2 migration: vends `TextKit2LayoutFragment`
/// instances configured with the per-paragraph code-block + blockquote
/// decoration data from the current `RenderSpec`.
///
/// Held strongly by the Coordinator (NSTextLayoutManager.delegate is weak,
/// same constraint as NSTextContentStorage.delegate).
///
/// Spec inputs are written by `TextKit2RenderApplicator.apply(_:to:)`. The
/// applicator then calls `NSTextLayoutManager.invalidateLayout(for:)` to
/// force re-vending so the new state takes effect.
@MainActor
final class TextKit2LayoutManagerDelegate: NSObject, @MainActor NSTextLayoutManagerDelegate {

  // MARK: - Spec inputs (written by TextKit2RenderApplicator)

  var blockquoteCharacterRanges: [RenderSpec.BlockquoteDecoration] = []

  /// Phase 2 bridging-layer: spec ranges that drive block-renderer hosts.
  /// When vending a fragment for the FIRST paragraph of one of these
  /// ranges, the delegate flips `containsBlockAttachment` on the fragment
  /// so its `renderingSurfaceBounds` widens to encompass the attachment
  /// view (which can be much taller than the host glyph line).
  var blockRendererSpecs: [BlockRendererSpec] = []

  /// Container width in points. Updated by `apply()` and on container
  /// resize. Vended fragments use this to draw a full-width background up
  /// to the right edge.
  var containerWidth: CGFloat = 0

  /// Diagnostic: incremented each time the delegate vends a fragment.
  /// Tests use this to confirm re-vending after spec changes.
  private(set) var fragmentBuildCount: Int = 0

  // MARK: - NSTextLayoutManagerDelegate

  func textLayoutManager(
    _ textLayoutManager: NSTextLayoutManager,
    textLayoutFragmentFor location: NSTextLocation,
    in textElement: NSTextElement
  ) -> NSTextLayoutFragment {
    fragmentBuildCount += 1

    let fragment = TextKit2LayoutFragment(
      textElement: textElement, range: textElement.elementRange)
    fragment.containerWidth = containerWidth

    // Resolve the paragraph's source range so we can match it against the
    // decoration ranges (which are in document offsets).
    if let elementRange = textElement.elementRange,
      let contentManager = textLayoutManager.textContentManager,
      let documentRange = contentManager.documentRange as NSTextRange?
    {
      let start = contentManager.offset(from: documentRange.location, to: elementRange.location)
      let length = contentManager.offset(from: elementRange.location, to: elementRange.endLocation)
      let paragraphSource = NSRange(location: start, length: length)

      // Blockquote: multiple nesting levels can overlap one paragraph; each
      // produces its own left-border line.
      let borders =
        blockquoteCharacterRanges
        .filter { rangesOverlap($0.range, paragraphSource) }
        .map { $0.xPosition }
      if !borders.isEmpty {
        fragment.blockquoteBorderXPositions = borders
      }

      // Block-renderer attachment: only the FIRST paragraph of a spec range
      // carries the U+FFFC attachment, so widen the surface bounds only on
      // that paragraph. Sibling paragraphs are hidden by the content
      // delegate's `shouldEnumerate` and never reach this hook.
      if let spec = blockRendererSpecs.first(where: {
        $0.range.location == paragraphSource.location
      }) {
        fragment.containsBlockAttachment = true
        fragment.blockAttachmentReservedHeight = spec.reservedHeight
      }
    }

    return fragment
  }

  // MARK: - Helpers

  private func rangesOverlap(_ a: NSRange, _ b: NSRange) -> Bool {
    NSIntersectionRange(a, b).length > 0
  }
}
