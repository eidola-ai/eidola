import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Bold/Italic Render Tests")
@MainActor
struct BoldItalicRenderTests {

  // MARK: - Bold

  @Test("Bold text produces bold font trait on content range")
  func boldFontTrait() {
    let text = "**bold**"
    let cursorRange = NSRange(location: 0, length: 0)  // cursor outside
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have a font trait for bold
    #expect(spec.fontTraits.count >= 1, "Should have at least one font trait")
    let boldTrait = spec.fontTraits.first { $0.trait == .boldFontMask }
    #expect(boldTrait != nil, "Should have a bold font trait")
    // Bold trait should apply to content "bold" (positions 2..6)
    #expect(boldTrait?.range == NSRange(location: 2, length: 4), "Bold trait should cover content range")
  }

  @Test("Bold delimiters hidden when cursor is outside")
  func boldDelimitersHiddenOutside() {
    let text = "hello **bold** world"
    // Cursor at position 0 (outside the bold)
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // ** at positions 6,7 and 12,13 should be hidden
    #expect(spec.hiddenIndexes.contains(6), "Opening ** first char should be hidden")
    #expect(spec.hiddenIndexes.contains(7), "Opening ** second char should be hidden")
    #expect(spec.hiddenIndexes.contains(12), "Closing ** first char should be hidden")
    #expect(spec.hiddenIndexes.contains(13), "Closing ** second char should be hidden")
  }

  @Test("Bold delimiters visible and dimmed when cursor is inside")
  func boldDelimitersVisibleInside() {
    let text = "**bold**"
    // Cursor at position 4 (inside the bold content)
    let cursorRange = NSRange(location: 4, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Delimiters should NOT be hidden
    #expect(!spec.hiddenIndexes.contains(0), "Opening ** should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(1), "Opening ** should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(6), "Closing ** should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(7), "Closing ** should not be hidden when cursor inside")

    // Should have temporary attributes for dimmed delimiters
    #expect(spec.temporaryAttributes.count >= 2, "Should have temp attrs for opening and closing delimiters")
  }

  // MARK: - Italic

  @Test("Italic text produces italic font trait on content range")
  func italicFontTrait() {
    let text = "*italic*"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let italicTrait = spec.fontTraits.first { $0.trait == .italicFontMask }
    #expect(italicTrait != nil, "Should have an italic font trait")
    // Content "italic" is at positions 1..7
    #expect(italicTrait?.range == NSRange(location: 1, length: 6), "Italic trait should cover content range")
  }

  @Test("Italic delimiters hidden when cursor is outside")
  func italicDelimitersHiddenOutside() {
    let text = "hello *italic* world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // * at position 6 and 13 should be hidden
    #expect(spec.hiddenIndexes.contains(6), "Opening * should be hidden")
    #expect(spec.hiddenIndexes.contains(13), "Closing * should be hidden")
  }

  @Test("Italic delimiters visible when cursor is inside")
  func italicDelimitersVisibleInside() {
    let text = "*italic*"
    let cursorRange = NSRange(location: 4, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "Opening * should not be hidden")
    #expect(!spec.hiddenIndexes.contains(7), "Closing * should not be hidden")
    #expect(spec.temporaryAttributes.count >= 2, "Should have temp attrs for delimiters")
  }

  // MARK: - Bold Italic

  @Test("Bold italic produces both traits")
  func boldItalicTraits() {
    let text = "***bold italic***"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let boldTrait = spec.fontTraits.first { $0.trait == .boldFontMask }
    let italicTrait = spec.fontTraits.first { $0.trait == .italicFontMask }
    #expect(boldTrait != nil, "Should have bold trait")
    #expect(italicTrait != nil, "Should have italic trait")
  }

  @Test("Bold italic delimiters hidden when cursor outside")
  func boldItalicDelimitersHidden() {
    // Use text with prefix so cursor at position 0 is clearly outside the construct.
    let text = "hello ***bold italic*** world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Both emphasis (*) and strong (**) delimiter characters should be hidden.
    // Emphasis has 1 char delimiters, strong has 2 char delimiters. Together
    // with overlapping positions in IndexSet, we get at least 3 per side = 6 total.
    // (Or 4 if they share all positions — either way, at least 4.)
    #expect(
      spec.hiddenIndexes.count >= 4,
      "Delimiter characters should be hidden, got \(spec.hiddenIndexes.count)")

    // Content "bold italic" should NOT be hidden
    let nsText = text as NSString
    let contentRange = nsText.range(of: "bold italic")
    for i in contentRange.location..<(contentRange.location + contentRange.length) {
      #expect(
        !spec.hiddenIndexes.contains(i),
        "Content position \(i) should not be hidden")
    }
  }

  // MARK: - Bold Inside Heading

  @Test("Bold inside heading produces bold trait and heading attributes")
  func boldInsideHeading() {
    let text = "# **bold heading**"
    let cursorRange = NSRange(location: 0, length: 0)  // cursor outside
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have heading styled range
    #expect(!spec.styledRanges.isEmpty, "Should have heading styled range")

    // Should have bold font trait
    let boldTrait = spec.fontTraits.first { $0.trait == .boldFontMask }
    #expect(boldTrait != nil, "Should have bold trait inside heading")
  }

  @Test("Bold inside heading hides both heading and bold delimiters when cursor outside")
  func boldInsideHeadingDelimitersHidden() {
    let text = "# **bold heading**\n\nBody"
    // Cursor in body text
    let cursorRange = NSRange(location: 22, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Heading delimiter "# " at 0,1 should be hidden
    #expect(spec.hiddenIndexes.contains(0), "# should be hidden")
    #expect(spec.hiddenIndexes.contains(1), "space after # should be hidden")

    // Bold delimiters ** at 2,3 and 16,17 should be hidden
    #expect(spec.hiddenIndexes.contains(2), "Opening ** first char should be hidden")
    #expect(spec.hiddenIndexes.contains(3), "Opening ** second char should be hidden")
    #expect(spec.hiddenIndexes.contains(16), "Closing ** first char should be hidden")
    #expect(spec.hiddenIndexes.contains(17), "Closing ** second char should be hidden")
  }

  // MARK: - Bold Italic Delimiter Coverage

  @Test("Bold italic hides ALL 3 opening and 3 closing asterisks when cursor outside")
  func boldItalicHidesAllAsterisks() {
    // Regression: ***text*** parsed as Emphasis(Strong(text)).
    // Both nodes share the same range. Strong's ** delimiters must be
    // offset inward by the Emphasis * delimiter width so that all 3
    // asterisks on each side are covered.
    let text = "***bold italic***"
    let cursorRange = NSRange(location: 0, length: 0)  // at start, before the construct
    // cursor at position 0 is at the node start, which counts as inside.
    // Use a document with prefix text so cursor is truly outside.
    let text2 = "hello ***bold italic*** world"
    let cursorRange2 = NSRange(location: 2, length: 0)  // clearly outside
    let spec = MarkdownRenderer.render(text: text2, cursorRange: cursorRange2)

    // Opening *** at positions 6,7,8
    #expect(spec.hiddenIndexes.contains(6), "Position 6 (first *) should be hidden")
    #expect(spec.hiddenIndexes.contains(7), "Position 7 (second *) should be hidden")
    #expect(spec.hiddenIndexes.contains(8), "Position 8 (third *) should be hidden")

    // Closing *** at positions 20,21,22
    #expect(spec.hiddenIndexes.contains(20), "Position 20 (first closing *) should be hidden")
    #expect(spec.hiddenIndexes.contains(21), "Position 21 (second closing *) should be hidden")
    #expect(spec.hiddenIndexes.contains(22), "Position 22 (third closing *) should be hidden")

    // Exactly 6 delimiter characters hidden (3 opening + 3 closing)
    #expect(spec.hiddenIndexes.count == 6, "Should hide exactly 6 asterisks, got \(spec.hiddenIndexes.count)")
  }

  @Test("Bold italic reveals ALL 3 asterisks when cursor inside")
  func boldItalicRevealsAllAsterisks() {
    let text = "***bold italic***"
    let cursorRange = NSRange(location: 5, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // No asterisks should be hidden
    #expect(spec.hiddenIndexes.isEmpty, "No characters should be hidden when cursor is inside")

    // All delimiter ranges should have temporary (dimmed) attributes
    // Emphasis has 2 delimiter ranges, Strong has 2 = 4 total
    #expect(
      spec.temporaryAttributes.count >= 4,
      "Should have temp attrs for all delimiter ranges, got \(spec.temporaryAttributes.count)")
  }

  // MARK: - Cursor at node boundaries

  @Test("Cursor at start of bold node reveals delimiters")
  func cursorAtStartOfBold() {
    let text = "**bold**"
    // Cursor at position 0 (at the opening **)
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Cursor is at the start of the node range, so it overlaps
    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at start of bold")
  }

  @Test("Cursor at end of bold node reveals delimiters")
  func cursorAtEndOfBold() {
    let text = "**bold**"
    // Cursor at position 8 (right after closing **)
    let cursorRange = NSRange(location: 8, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Cursor is at the end of the node, should be considered inside
    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at end of bold")
  }

  @Test("Cursor just outside bold node hides delimiters")
  func cursorOutsideBoldNode() {
    let text = "text **bold** more"
    // Cursor at position 0 (clearly outside)
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.isEmpty, "Delimiters should be hidden when cursor outside bold")
  }
}
