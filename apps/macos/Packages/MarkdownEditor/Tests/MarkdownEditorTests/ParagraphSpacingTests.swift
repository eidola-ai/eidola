import AppKit
import Testing

@testable import MarkdownEditor

/// Tests for paragraph spacing stability when cursor enters/leaves constructs.
///
/// The core bug: when the cursor leaves a construct whose delimiter starts at the
/// beginning of a paragraph line, the paragraph's top spacing shrinks because TextKit
/// computes line fragment metrics from null glyphs differently than visible glyphs.
@Suite("Paragraph Spacing Stability Tests")
@MainActor
struct ParagraphSpacingTests {

  // MARK: - Heading Spacing Stability

  @Test("Heading spacing visual snapshot stable when cursor moves in and out")
  func headingSpacingVisualSnapshot() {
    let markdown = "Some body text\n### Features\nMore body"
    let initial = EditorState(markdown: markdown, selection: .cursor(22))  // inside "Features"
    let events: [EditorEvent] = [
      .setSelection(.cursor(5)),   // move to body (heading hidden)
      .setSelection(.cursor(22)),  // back inside heading
    ]
    let results = EditorTestHarness.run(
      name: "heading-spacing-stability",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    // step 0 (cursor inside heading) and step 2 (cursor back inside) should match
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce identical rendering")
  }

  @Test("Heading line fragment rect is stable via layout manager")
  func headingLineFragmentRectStable() {
    // Use the layout manager directly to check line fragment positioning
    let markdown = "Some body text\n### Features"
    let size = NSSize(width: 600, height: 200)

    let components = MarkdownTextViewFactory.create(size: size)
    let layoutManager = components.layoutManager

    // Render with cursor inside heading (delimiters visible)
    SnapshotCapture.apply(text: markdown, cursorPosition: 19, to: components)
    layoutManager.ensureLayout(
      forCharacterRange: NSRange(location: 0, length: (markdown as NSString).length))

    let glyphIndex = layoutManager.glyphIndexForCharacter(at: 19)
    var insideRange = NSRange()
    let insideRect = layoutManager.lineFragmentRect(
      forGlyphAt: glyphIndex, effectiveRange: &insideRange)
    let insideUsed = layoutManager.lineFragmentUsedRect(
      forGlyphAt: glyphIndex, effectiveRange: nil)

    // Render with cursor outside heading (delimiters hidden)
    SnapshotCapture.apply(text: markdown, cursorPosition: 5, to: components)
    layoutManager.ensureLayout(
      forCharacterRange: NSRange(location: 0, length: (markdown as NSString).length))

    let glyphIndex2 = layoutManager.glyphIndexForCharacter(at: 19)
    var outsideRange = NSRange()
    let outsideRect = layoutManager.lineFragmentRect(
      forGlyphAt: glyphIndex2, effectiveRange: &outsideRange)
    let outsideUsed = layoutManager.lineFragmentUsedRect(
      forGlyphAt: glyphIndex2, effectiveRange: nil)

    let yShift = abs(insideRect.origin.y - outsideRect.origin.y)
    #expect(
      yShift < 1.0,
      "Heading line fragment Y should not shift. Inside: \(insideRect.origin.y), Outside: \(outsideRect.origin.y), Shift: \(yShift)"
    )

    // The heading line fragment should also have the same height (including paragraphSpacingBefore)
    let heightShift = abs(insideRect.size.height - outsideRect.size.height)
    #expect(
      heightShift < 1.0,
      "Heading line fragment height should not shift. Inside: \(insideRect.size.height), Outside: \(outsideRect.size.height), Shift: \(heightShift)"
    )

    // The used rect's Y position (which includes paragraphSpacingBefore offset) should be consistent
    let usedYShift = abs(insideUsed.origin.y - outsideUsed.origin.y)
    #expect(
      usedYShift < 1.0,
      "Heading used rect Y should not shift. Inside: \(insideUsed.origin.y), Outside: \(outsideUsed.origin.y), Shift: \(usedYShift)"
    )
  }

  // MARK: - Italic at Line Start Spacing Stability

  @Test("Italic at line start Y position does not shift when cursor leaves")
  func italicAtLineStartYPositionStable() {
    // Document: italic text at the start of line 2
    let markdown = "Normal text\n*This is italic*"
    let size = NSSize(width: 600, height: 200)

    let components = MarkdownTextViewFactory.create(size: size)
    let layoutManager = components.layoutManager

    // Cursor inside italic
    SnapshotCapture.apply(text: markdown, cursorPosition: 15, to: components)
    layoutManager.ensureLayout(
      forCharacterRange: NSRange(location: 0, length: (markdown as NSString).length))
    // char at index 13 is 'T' in "This"
    let glyph1 = layoutManager.glyphIndexForCharacter(at: 13)
    var range1 = NSRange()
    let rect1 = layoutManager.lineFragmentRect(forGlyphAt: glyph1, effectiveRange: &range1)

    // Cursor outside italic
    SnapshotCapture.apply(text: markdown, cursorPosition: 5, to: components)
    layoutManager.ensureLayout(
      forCharacterRange: NSRange(location: 0, length: (markdown as NSString).length))
    let glyph2 = layoutManager.glyphIndexForCharacter(at: 13)
    var range2 = NSRange()
    let rect2 = layoutManager.lineFragmentRect(forGlyphAt: glyph2, effectiveRange: &range2)

    let yShift = abs(rect1.origin.y - rect2.origin.y)
    #expect(
      yShift < 1.0,
      "Italic line fragment Y should not shift. Inside: \(rect1.origin.y), Outside: \(rect2.origin.y), Shift: \(yShift)"
    )
  }
}
