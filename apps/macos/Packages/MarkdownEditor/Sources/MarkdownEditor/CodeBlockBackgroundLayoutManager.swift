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
  nonisolated(unsafe) var codeBlockCharacterRanges: [RenderSpec.CodeBlockRange] = []

  /// Character ranges that should receive a left border for blockquote indication.
  /// Set by `RenderApplicator` after each render pass.
  nonisolated(unsafe) var blockquoteCharacterRanges: [RenderSpec.BlockquoteRange] = []

  /// The background color to draw behind code block lines.
  nonisolated(unsafe) var codeBlockBackgroundColor: NSColor = .quaternaryLabelColor
    .withAlphaComponent(0.5)

  /// The color to draw blockquote left borders.
  nonisolated(unsafe) var blockquoteBorderColor: NSColor = .tertiaryLabelColor

  /// The indent per blockquote depth level (must match MarkdownStyle.blockquoteIndent).
  nonisolated(unsafe) var blockquoteIndent: CGFloat = 20

  /// Width of the blockquote left border line.
  nonisolated(unsafe) var blockquoteBorderWidth: CGFloat = 2

  /// Left padding before the first blockquote border line.
  nonisolated(unsafe) var blockquoteBorderLeftPadding: CGFloat = 6

  override func drawBackground(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
    // Draw standard backgrounds first (for non-code-block ranges).
    super.drawBackground(forGlyphRange: glyphsToShow, at: origin)

    // Draw full-width backgrounds for code block lines.
    if !codeBlockCharacterRanges.isEmpty,
      let textContainer = textContainers.first
    {
      let containerWidth = textContainer.containerSize.width
      if containerWidth > 0, containerWidth < CGFloat.greatestFiniteMagnitude {
        codeBlockBackgroundColor.setFill()

        for codeRange in codeBlockCharacterRanges {
          let codeGlyphRange = glyphRange(forCharacterRange: codeRange.range, actualCharacterRange: nil)

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
            // Draw a full-width rectangle at this line's vertical position,
            // offset by the code block's base indent (for list nesting).
            var rect = lineFragmentRect
            rect.origin.x = codeRange.baseIndent
            rect.size.width = containerWidth - codeRange.baseIndent
            rect.origin.x += origin.x
            rect.origin.y += origin.y
            NSBezierPath.fill(rect)
          }
        }
      }
    }

    // Draw left borders for blockquote ranges.
    guard !blockquoteCharacterRanges.isEmpty else { return }

    for bqRange in blockquoteCharacterRanges {
      let bqGlyphRange = glyphRange(forCharacterRange: bqRange.range, actualCharacterRange: nil)

      // Only draw if this blockquote overlaps the glyphs we're asked to draw.
      let overlapStart = max(bqGlyphRange.location, glyphsToShow.location)
      let overlapEnd = min(
        bqGlyphRange.location + bqGlyphRange.length,
        glyphsToShow.location + glyphsToShow.length)
      guard overlapStart < overlapEnd else { continue }

      // Enumerate line fragments within the visible portion of this blockquote.
      // Use the overlap range so hidden-glyph lines that still have line fragments
      // are included.
      enumerateLineFragments(
        forGlyphRange: NSRange(location: overlapStart, length: overlapEnd - overlapStart)
      ) { lineFragmentRect, _, _, _, _ in
        // Draw ONE vertical border line at this depth's position. Each blockquote
        // depth level has its own entry in blockquoteCharacterRanges, so parent
        // levels draw their own borders — no need to re-draw them here.
        let xPosition =
          bqRange.listBaseIndent + self.blockquoteBorderLeftPadding + self.blockquoteIndent * CGFloat(bqRange.depth - 1)
        self.blockquoteBorderColor.setFill()
        let borderRect = NSRect(
          x: xPosition + origin.x,
          y: lineFragmentRect.origin.y + origin.y,
          width: self.blockquoteBorderWidth,
          height: lineFragmentRect.size.height)
        NSBezierPath.fill(borderRect)
      }
    }
  }
}
