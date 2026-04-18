import AppKit

/// Custom NSLayoutManager subclass that draws full-width backgrounds for code block lines.
///
/// Standard `.backgroundColor` only paints behind rendered glyphs. When fence characters
/// are hidden (`.null` glyph property), their glyphs have zero width, so the background
/// only covers a tiny area. This subclass draws a full-width rectangle for every line
/// that falls within a code block range, producing a uniform "box" appearance.
@MainActor
final class CodeBlockBackgroundLayoutManager: NSLayoutManager {
  /// Character ranges that should receive full-width code block background.
  /// Set by `RenderApplicator` after each render pass.
  nonisolated(unsafe) var codeBlockCharacterRanges: [NSRange] = []

  /// The background color to draw behind code block lines.
  nonisolated(unsafe) var codeBlockBackgroundColor: NSColor = .quaternaryLabelColor
    .withAlphaComponent(0.5)

  override func drawBackground(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
    // Draw standard backgrounds first (for non-code-block ranges).
    super.drawBackground(forGlyphRange: glyphsToShow, at: origin)

    // Draw full-width backgrounds for code block lines.
    guard !codeBlockCharacterRanges.isEmpty,
      let textContainer = textContainers.first
    else { return }

    let containerWidth = textContainer.containerSize.width
    guard containerWidth > 0, containerWidth < CGFloat.greatestFiniteMagnitude else { return }

    codeBlockBackgroundColor.setFill()

    for codeRange in codeBlockCharacterRanges {
      let codeGlyphRange = glyphRange(forCharacterRange: codeRange, actualCharacterRange: nil)

      // Only draw if this code block overlaps the glyphs we're asked to draw.
      let overlapStart = max(codeGlyphRange.location, glyphsToShow.location)
      let overlapEnd = min(
        codeGlyphRange.location + codeGlyphRange.length,
        glyphsToShow.location + glyphsToShow.length)
      guard overlapStart < overlapEnd else { continue }

      // Enumerate line fragment rects within the code block's glyph range.
      // Each line fragment rect spans the full text container width (for non-hidden lines)
      // or a narrow width (for hidden-glyph lines). We draw a full-width rect for all.
      enumerateLineFragments(
        forGlyphRange: NSRange(location: overlapStart, length: overlapEnd - overlapStart)
      ) { lineFragmentRect, _, _, _, _ in
        // Draw a full-width rectangle at this line's vertical position.
        var rect = lineFragmentRect
        rect.origin.x = 0
        rect.size.width = containerWidth
        rect.origin.x += origin.x
        rect.origin.y += origin.y
        NSBezierPath.fill(rect)
      }
    }
  }
}
