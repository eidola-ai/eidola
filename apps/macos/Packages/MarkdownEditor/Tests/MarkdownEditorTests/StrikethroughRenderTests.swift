import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Strikethrough Render Tests")
@MainActor
struct StrikethroughRenderTests {

  // MARK: - Strikethrough attribute

  @Test("Strikethrough text has strikethrough style on content range")
  func strikethroughAttribute() {
    let text = "~~struck~~"
    let cursorRange = NSRange(location: 0, length: 0)  // cursor outside (at node start = inside)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have a styled range with strikethrough
    let strikethroughStyled = spec.styledRanges.first { styled in
      styled.attributes[.strikethroughStyle] != nil
    }
    // When cursor is at position 0 (node start), cursor is inside, so delimiters are visible
    // and strikethrough attribute is still applied to content.
    // Content "struck" is at positions 2..8
    #expect(strikethroughStyled != nil, "Should have strikethrough styled range")
    #expect(
      strikethroughStyled?.range == NSRange(location: 2, length: 6),
      "Strikethrough style should cover content range")
  }

  @Test("Strikethrough attribute present when cursor outside")
  func strikethroughAttributeOutside() {
    let text = "hello ~~struck~~ world"
    let cursorRange = NSRange(location: 0, length: 0)  // clearly outside
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let strikethroughStyled = spec.styledRanges.first { styled in
      styled.attributes[.strikethroughStyle] != nil
    }
    #expect(strikethroughStyled != nil, "Should have strikethrough styled range")
    // Content "struck" is at positions 8..14
    #expect(
      strikethroughStyled?.range == NSRange(location: 8, length: 6),
      "Strikethrough style should cover content range")
  }

  // MARK: - Delimiter hiding/revealing

  @Test("Strikethrough delimiters hidden when cursor is outside")
  func delimitersHiddenOutside() {
    let text = "hello ~~struck~~ world"
    let cursorRange = NSRange(location: 0, length: 0)  // clearly outside
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // ~~ at positions 6,7 and 14,15 should be hidden
    #expect(spec.hiddenIndexes.contains(6), "Opening ~~ first char should be hidden")
    #expect(spec.hiddenIndexes.contains(7), "Opening ~~ second char should be hidden")
    #expect(spec.hiddenIndexes.contains(14), "Closing ~~ first char should be hidden")
    #expect(spec.hiddenIndexes.contains(15), "Closing ~~ second char should be hidden")

    // Content should NOT be hidden
    for i in 8...13 {
      #expect(!spec.hiddenIndexes.contains(i), "Content position \(i) should not be hidden")
    }
  }

  @Test("Strikethrough delimiters visible and dimmed when cursor is inside")
  func delimitersVisibleInside() {
    let text = "~~struck~~"
    let cursorRange = NSRange(location: 5, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Delimiters should NOT be hidden
    #expect(!spec.hiddenIndexes.contains(0), "Opening ~~ should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(1), "Opening ~~ should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(8), "Closing ~~ should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(9), "Closing ~~ should not be hidden when cursor inside")

    // Should have temporary attributes for dimmed delimiters
    #expect(
      spec.temporaryAttributes.count >= 2,
      "Should have temp attrs for opening and closing delimiters")
  }

  // MARK: - Cursor at boundaries

  @Test("Cursor at start of strikethrough node reveals delimiters")
  func cursorAtStart() {
    let text = "~~struck~~"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Cursor at node start counts as inside
    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at start")
  }

  @Test("Cursor at end of strikethrough node reveals delimiters")
  func cursorAtEnd() {
    let text = "~~struck~~"
    let cursorRange = NSRange(location: 10, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Cursor at node end counts as inside
    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at end")
  }

  @Test("Cursor just outside strikethrough hides delimiters")
  func cursorOutside() {
    let text = "text ~~struck~~ more"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.isEmpty, "Delimiters should be hidden when cursor outside")
  }

  // MARK: - Strikethrough combined with other formatting

  @Test("Strikethrough inside heading works")
  func strikethroughInsideHeading() {
    let text = "# ~~struck heading~~"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have heading styled range
    #expect(!spec.styledRanges.isEmpty, "Should have styled ranges")

    // Should have strikethrough attribute
    let hasStrikethrough = spec.styledRanges.contains { styled in
      styled.attributes[.strikethroughStyle] != nil
    }
    #expect(hasStrikethrough, "Should have strikethrough style inside heading")
  }

  @Test("Bold strikethrough produces both traits")
  func boldStrikethrough() {
    let text = "**~~bold struck~~**"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have bold trait
    let boldTrait = spec.fontTraits.first { $0.trait == .boldFontMask }
    #expect(boldTrait != nil, "Should have bold trait")

    // Should have strikethrough attribute
    let hasStrikethrough = spec.styledRanges.contains { styled in
      styled.attributes[.strikethroughStyle] != nil
    }
    #expect(hasStrikethrough, "Should have strikethrough style")
  }

  // MARK: - Single tilde variant

  @Test("Single tilde ~word~ has correct delimiter width and content range")
  func singleTildeStrikethrough() {
    let text = "hello ~word~ world"
    let cursorRange = NSRange(location: 0, length: 0)  // cursor outside
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have strikethrough attribute on "word" (positions 7-10)
    let strikethroughStyled = spec.styledRanges.first { styled in
      styled.attributes[.strikethroughStyle] != nil
    }
    #expect(strikethroughStyled != nil, "Single ~ should produce strikethrough")
    if let styled = strikethroughStyled {
      #expect(styled.range.location == 7, "Content should start at position 7 ('w' of 'word')")
      #expect(styled.range.length == 4, "Content should be 4 chars ('word')")
    }

    // Delimiters at positions 6 (~) and 11 (~) should be hidden
    #expect(spec.hiddenIndexes.contains(6), "Opening ~ should be hidden")
    #expect(spec.hiddenIndexes.contains(11), "Closing ~ should be hidden")

    // Content should NOT be hidden
    for i in 7...10 {
      #expect(!spec.hiddenIndexes.contains(i), "Content char at \(i) should not be hidden")
    }
  }

  @Test("Single tilde ~word~ with cursor inside reveals single ~ delimiters")
  func singleTildeCursorInside() {
    let text = "hello ~word~ world"
    let cursorRange = NSRange(location: 8, length: 0)  // inside "word"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Delimiters should NOT be hidden
    #expect(!spec.hiddenIndexes.contains(6), "Opening ~ should be visible when cursor inside")
    #expect(!spec.hiddenIndexes.contains(11), "Closing ~ should be visible when cursor inside")

    // Should have temporary attributes for dimmed delimiters
    #expect(!spec.temporaryAttributes.isEmpty, "Should have dimmed delimiter attributes")
  }

  @Test("Double tilde ~~word~~ still works correctly alongside single")
  func doubleTildeStillWorks() {
    let text = "~single~ and ~~double~~"
    let cursorRange = NSRange(location: 23, length: 0)  // cursor at end, outside both
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Both should have strikethrough
    let strikethroughRanges = spec.styledRanges.filter { styled in
      styled.attributes[.strikethroughStyle] != nil
    }
    #expect(strikethroughRanges.count == 2, "Both single and double tilde should produce strikethrough")
  }
}
