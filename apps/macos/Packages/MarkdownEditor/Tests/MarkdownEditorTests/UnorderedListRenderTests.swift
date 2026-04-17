import AppKit
import Testing

@testable import MarkdownEditor

/// Functional tests for unordered list rendering: RenderSpec properties.
@Suite("Unordered List Render Tests")
@MainActor
struct UnorderedListRenderTests {

  // MARK: - Bullet substitution when cursor outside

  @Test("List item shows bullet glyph when cursor is outside")
  func bulletGlyphWhenCursorOutside() {
    let text = "- Hello\n\nBody"
    // Cursor in body (position 10)
    let cursorRange = NSRange(location: 10, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Position 0 (the '-') should be in bulletIndexes
    #expect(spec.bulletIndexes.contains(0), "Marker char should be a bullet glyph")
    // Position 1 (space) stays visible for spacing between bullet and content
    #expect(!spec.hiddenIndexes.contains(1), "Space after marker should be visible for spacing")
    // Content should not be hidden or bullet
    #expect(!spec.hiddenIndexes.contains(2), "Content should not be hidden")
    #expect(!spec.bulletIndexes.contains(2), "Content should not be bullet")
  }

  @Test("List item with * marker shows bullet when cursor outside")
  func bulletGlyphWithStarMarker() {
    let text = "* Hello\n\nBody"
    let cursorRange = NSRange(location: 10, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.bulletIndexes.contains(0), "Star marker should become bullet")
    #expect(!spec.hiddenIndexes.contains(1), "Space after star should be visible")
  }

  @Test("List item with + marker shows bullet when cursor outside")
  func bulletGlyphWithPlusMarker() {
    let text = "+ Hello\n\nBody"
    let cursorRange = NSRange(location: 10, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.bulletIndexes.contains(0), "Plus marker should become bullet")
    #expect(!spec.hiddenIndexes.contains(1), "Space after plus should be visible")
  }

  // MARK: - Delimiter dimmed when cursor inside

  @Test("List item delimiter dimmed when cursor is inside")
  func delimiterDimmedWhenCursorInside() {
    let text = "- Hello"
    let cursorRange = NSRange(location: 4, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Marker should NOT be in bulletIndexes or hiddenIndexes
    #expect(!spec.bulletIndexes.contains(0), "Marker should not be bullet when cursor inside")
    #expect(!spec.hiddenIndexes.contains(0), "Marker char should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(1), "Space should not be hidden when cursor inside")

    // Should have temporary attributes for dimmed delimiter
    #expect(
      spec.temporaryAttributes.count >= 1,
      "Should have temp attrs for dimmed delimiter")
  }

  @Test("Cursor at start of list item reveals delimiter")
  func cursorAtStartRevealsDelimiter() {
    let text = "- Hello"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.bulletIndexes.isEmpty, "No bullets when cursor at start")
    #expect(spec.hiddenIndexes.isEmpty, "Nothing hidden when cursor at start")
  }

  @Test("Cursor at end of list item reveals delimiter")
  func cursorAtEndRevealsDelimiter() {
    let text = "- Hello"
    let cursorRange = NSRange(location: 7, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.bulletIndexes.isEmpty, "No bullets when cursor at end of list item")
    #expect(spec.hiddenIndexes.isEmpty, "Nothing hidden when cursor at end of list item")
  }

  // MARK: - Multiple list items

  @Test("Multiple list items each get bullet treatment when cursor outside")
  func multipleListItemsBullets() {
    let text = "- Item 1\n- Item 2\n- Item 3\n\nBody"
    // Cursor in body
    let cursorRange = NSRange(location: 30, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Each list item's marker char should be a bullet
    #expect(spec.bulletIndexes.contains(0), "First item marker should be bullet")
    #expect(spec.bulletIndexes.contains(9), "Second item marker should be bullet")
    #expect(spec.bulletIndexes.contains(18), "Third item marker should be bullet")

    // Each list item's space should be visible (spacing between bullet and content)
    #expect(!spec.hiddenIndexes.contains(1), "First item space should be visible")
    #expect(!spec.hiddenIndexes.contains(10), "Second item space should be visible")
    #expect(!spec.hiddenIndexes.contains(19), "Third item space should be visible")
  }

  @Test("Cursor on one list item reveals only that item's delimiter")
  func cursorOnOneItemRevealsOnlyThat() {
    let text = "- Item 1\n- Item 2\n- Item 3\n\nBody"
    // Cursor inside second item (position 14 = inside "Item 2")
    let cursorRange = NSRange(location: 14, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // First and third items should still have bullets
    #expect(spec.bulletIndexes.contains(0), "First item should have bullet")
    #expect(spec.bulletIndexes.contains(18), "Third item should have bullet")

    // Second item should NOT have bullet (cursor inside)
    #expect(!spec.bulletIndexes.contains(9), "Second item should not have bullet")
    #expect(!spec.hiddenIndexes.contains(9), "Second item marker should not be hidden")
    #expect(!spec.hiddenIndexes.contains(10), "Second item space should not be hidden")
  }

  // MARK: - Indentation attributes

  @Test("List item has paragraph indentation attributes")
  func listItemHasIndentation() {
    let text = "- Hello\n\nBody"
    let cursorRange = NSRange(location: 10, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have styled ranges for the list item
    let listStyled = spec.styledRanges.first { styled in
      styled.range.location == 0
    }
    #expect(listStyled != nil, "List item should have styled range")

    if let styled = listStyled {
      let paragraphStyle = styled.attributes[.paragraphStyle] as? NSParagraphStyle
      #expect(paragraphStyle != nil, "Should have paragraph style")
      #expect(
        paragraphStyle!.headIndent > 0,
        "headIndent should be > 0 for indentation, got \(paragraphStyle!.headIndent)")
      #expect(
        paragraphStyle!.firstLineHeadIndent > 0,
        "firstLineHeadIndent should be > 0")
    }
  }

  // MARK: - List with inline formatting

  @Test("Bold inside list item works correctly")
  func boldInsideListItem() {
    let text = "- **bold** text\n\nBody"
    let cursorRange = NSRange(location: 19, length: 0)  // in body
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // List marker should be bullet
    #expect(spec.bulletIndexes.contains(0), "List marker should be bullet")
    // Bold delimiters should be hidden
    #expect(spec.hiddenIndexes.contains(2), "Opening ** first char should be hidden")
    #expect(spec.hiddenIndexes.contains(3), "Opening ** second char should be hidden")
    // Bold content should have bold trait
    let boldTrait = spec.fontTraits.first { $0.trait == .boldFontMask }
    #expect(boldTrait != nil, "Should have bold trait inside list item")
  }
}
