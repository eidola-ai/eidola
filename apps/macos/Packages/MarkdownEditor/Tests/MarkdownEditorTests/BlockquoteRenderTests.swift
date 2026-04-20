import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Blockquote Render Tests")
@MainActor
struct BlockquoteRenderTests {

  // MARK: - Attributes

  @Test("Blockquote applies secondary label color")
  func blockquoteSecondaryColor() {
    let text = "> Hello world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let blockquoteStyled = spec.styledRanges.first {
      $0.attributes[.foregroundColor] as? NSColor == .secondaryLabelColor
    }
    #expect(
      blockquoteStyled != nil,
      "Blockquote should have secondary label color")
  }

  @Test("Blockquote applies paragraph indentation when cursor is outside")
  func blockquoteIndentationOutside() {
    let text = "> Hello world\n\nBody"
    let cursorRange = NSRange(location: 16, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let blockquoteStyled = spec.styledRanges.first {
      ($0.attributes[.paragraphStyle] as? NSParagraphStyle)?.headIndent ?? 0 > 0
    }
    #expect(
      blockquoteStyled != nil,
      "Blockquote should have paragraph style with head indent when cursor outside")

    if let styled = blockquoteStyled {
      let ps = styled.attributes[.paragraphStyle] as! NSParagraphStyle
      // firstLineHeadIndent is 0 because the transparent > glyph + kern provides the indent
      #expect(ps.firstLineHeadIndent == 0, "firstLineHeadIndent should be 0 when cursor outside (transparent > glyph)")
      #expect(ps.headIndent > 0, "headIndent should be > 0 when cursor outside")
    }
  }

  @Test("Blockquote has zero paragraph indentation when cursor is inside")
  func blockquoteIndentationInside() {
    let text = "> Hello world"
    let cursorRange = NSRange(location: 4, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let blockquoteStyled = spec.styledRanges.first {
      ($0.attributes[.paragraphStyle] as? NSParagraphStyle) != nil
        && ($0.attributes[.foregroundColor] as? NSColor) == .secondaryLabelColor
    }
    #expect(
      blockquoteStyled != nil,
      "Blockquote should have paragraph style when cursor inside")

    if let styled = blockquoteStyled {
      let ps = styled.attributes[.paragraphStyle] as! NSParagraphStyle
      #expect(ps.firstLineHeadIndent == 0, "firstLineHeadIndent should be 0 when cursor inside (> stays visible)")
      #expect(ps.headIndent > 0, "headIndent should stay positive so wrapped lines align after the visible > prefix")
    }
  }

  // MARK: - Delimiter Hiding

  @Test("Blockquote > prefix hidden when cursor is outside")
  func prefixHiddenOutside() {
    let text = "> Hello\n\nBody"
    let cursorRange = NSRange(location: 10, length: 0)  // cursor in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // > at position 0 is kept as a transparent glyph (not hidden); space at position 1 is hidden
    #expect(!spec.hiddenIndexes.contains(0), "> char should not be hidden (it's transparent, same kern)")
    #expect(spec.hiddenIndexes.contains(1), "space after > should be hidden when cursor outside")

    // Content should not be hidden
    #expect(!spec.hiddenIndexes.contains(2), "Content should not be hidden")
  }

  @Test("Blockquote > prefix visible and dimmed when cursor is inside")
  func prefixVisibleInside() {
    let text = "> Hello"
    let cursorRange = NSRange(location: 4, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // > character should NOT be hidden (it has kern to fill blockquoteIndent)
    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden when cursor inside")
    // space after > IS hidden — kern on > provides the spacing instead
    #expect(spec.hiddenIndexes.contains(1), "space after > should be hidden (kern provides spacing)")

    // Should have temporary attributes for dimmed delimiter
    #expect(
      !spec.temporaryAttributes.isEmpty,
      "Should have dimmed delimiter temp attrs")
  }

  @Test("Blockquote border ranges populated when cursor is outside")
  func blockquoteBorderRangesOutside() {
    let text = "> Hello\n\nBody"
    let cursorRange = NSRange(location: 10, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)
    #expect(!spec.blockquoteCharacterRanges.isEmpty,
      "Should have blockquote border ranges when cursor is outside")
    #expect((spec.blockquoteCharacterRanges.first?.xPosition ?? 0) > 0,
      "Should have a positive x position")
  }

  @Test("Blockquote border ranges empty when cursor is inside")
  func blockquoteBorderRangesInside() {
    let text = "> Hello"
    let cursorRange = NSRange(location: 4, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)
    #expect(spec.blockquoteCharacterRanges.isEmpty,
      "Should not have blockquote border ranges when cursor is inside")
  }

  @Test("Blockquote > prefix visible when cursor is at start")
  func prefixVisibleAtStart() {
    let text = "> Hello"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden")
    // space after > is hidden (kern provides spacing)
    #expect(!spec.hiddenIndexes.contains(2), "Content should not be hidden")
  }

  @Test("Blockquote > prefix visible when cursor is at end")
  func prefixVisibleAtEnd() {
    let text = "> Hello"
    let cursorRange = NSRange(location: 7, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden")
    // space after > is hidden (kern provides spacing)
    #expect(!spec.hiddenIndexes.contains(2), "Content should not be hidden")
  }

  // MARK: - Multi-line blockquotes

  @Test("Multi-line blockquote hides all > prefixes when cursor outside")
  func multiLineHidesAllPrefixes() {
    let text = "> Line one\n> Line two\n\nBody"
    // Cursor in Body
    let cursorRange = NSRange(location: 24, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // First line: > at 0 is transparent (not hidden), space at 1 is hidden
    #expect(!spec.hiddenIndexes.contains(0), "First > should not be hidden (transparent glyph)")
    #expect(spec.hiddenIndexes.contains(1), "First space should be hidden")

    // Second line: > at 11 is transparent (not hidden), space at 12 is hidden
    #expect(!spec.hiddenIndexes.contains(11), "Second > should not be hidden (transparent glyph)")
    #expect(spec.hiddenIndexes.contains(12), "Second space should be hidden")

    // Content not hidden
    #expect(!spec.hiddenIndexes.contains(2), "First line content not hidden")
    #expect(!spec.hiddenIndexes.contains(13), "Second line content not hidden")
  }

  @Test("Multi-line blockquote reveals all > prefixes when cursor inside any line")
  func multiLineRevealsAllPrefixes() {
    let text = "> Line one\n> Line two\n\nBody"
    // Cursor inside second line content
    let cursorRange = NSRange(location: 15, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // > characters should not be hidden; spaces after > ARE hidden (kern provides spacing)
    #expect(!spec.hiddenIndexes.contains(0), "First > should not be hidden")
    #expect(spec.hiddenIndexes.contains(1), "First space after > should be hidden (kern provides spacing)")
    #expect(!spec.hiddenIndexes.contains(11), "Second > should not be hidden")
    #expect(spec.hiddenIndexes.contains(12), "Second space after > should be hidden (kern provides spacing)")

    // Should have temp attrs for dimming
    #expect(
      !spec.temporaryAttributes.isEmpty,
      "Should have dimmed delimiter temp attrs")
  }

  @Test("Multi-line blockquote reveals when cursor on first line")
  func multiLineRevealsFromFirstLine() {
    let text = "> Line one\n> Line two\n\nBody"
    // Cursor inside first line content
    let cursorRange = NSRange(location: 5, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // No prefixes should be hidden
    #expect(!spec.hiddenIndexes.contains(0), "First > should not be hidden")
    #expect(!spec.hiddenIndexes.contains(11), "Second > should not be hidden")
  }

  // MARK: - Content not hidden

  @Test("Blockquote content is not hidden")
  func contentNotHidden() {
    let text = "> Hello world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Content "Hello world" at positions 2..12 should not be hidden
    for i in 2...12 {
      #expect(!spec.hiddenIndexes.contains(i), "Content position \(i) should not be hidden")
    }
  }

  // MARK: - Blockquote with inline formatting

  @Test("Bold inside blockquote works correctly")
  func boldInsideBlockquote() {
    let text = "> **bold** text\n\nBody"
    let cursorRange = NSRange(location: 18, length: 0)  // in body
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // > prefix: > is transparent (not hidden), space is hidden
    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden (transparent glyph)")
    #expect(spec.hiddenIndexes.contains(1), "space should be hidden")
    // Bold delimiters should be hidden
    #expect(spec.hiddenIndexes.contains(2), "Opening ** first char should be hidden")
    #expect(spec.hiddenIndexes.contains(3), "Opening ** second char should be hidden")
    // Bold content should have bold trait
    let boldTrait = spec.fontTraits.first { $0.trait == .boldFontMask }
    #expect(boldTrait != nil, "Should have bold trait inside blockquote")
  }

  @Test("Italic inside blockquote works correctly")
  func italicInsideBlockquote() {
    let text = "> *italic* text\n\nBody"
    let cursorRange = NSRange(location: 19, length: 0)  // in body
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // > prefix: > is transparent (not hidden), space after is hidden
    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden (transparent glyph)")
    // Italic delimiters should be hidden
    #expect(spec.hiddenIndexes.contains(2), "Opening * should be hidden")
    // Italic content should have italic trait
    let italicTrait = spec.fontTraits.first { $0.trait == .italicFontMask }
    #expect(italicTrait != nil, "Should have italic trait inside blockquote")
  }

  // MARK: - Cursor at many boundary positions

  @Test("Cursor just outside blockquote hides > prefix")
  func cursorJustOutsideHidesPrefix() {
    let text = "> Hello\n\nBody"
    // Cursor at position 8 (blank line, just outside)
    let cursorRange = NSRange(location: 8, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden when cursor is just outside (transparent glyph)")
  }

  @Test("Cursor on > prefix reveals delimiters")
  func cursorOnPrefixReveals() {
    let text = "> Hello\n\nBody"
    // Cursor at position 1 (on space after >)
    let cursorRange = NSRange(location: 1, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden")
    // space after > is hidden (kern on > provides the spacing)
    #expect(!spec.hiddenIndexes.contains(2), "Content should not be hidden")
  }
}
