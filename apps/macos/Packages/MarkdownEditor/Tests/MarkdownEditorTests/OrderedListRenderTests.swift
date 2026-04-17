import AppKit
import Testing

@testable import MarkdownEditor

/// Functional tests for ordered list rendering: RenderSpec properties.
@Suite("Ordered List Render Tests")
@MainActor
struct OrderedListRenderTests {

  // MARK: - No bullet substitution or hiding for ordered lists

  @Test("Ordered list item has no bulletIndexes or hiddenIndexes when cursor outside")
  func noBulletOrHiddenWhenCursorOutside() {
    let text = "1. Hello\n\nBody"
    // Cursor in body (position 11)
    let cursorRange = NSRange(location: 11, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Ordered list markers should NOT be in bulletIndexes
    #expect(!spec.bulletIndexes.contains(0), "Ordered marker should not be a bullet glyph")
    #expect(!spec.bulletIndexes.contains(1), "Ordered marker dot should not be a bullet glyph")
    #expect(!spec.bulletIndexes.contains(2), "Ordered marker space should not be a bullet glyph")

    // Ordered list markers should NOT be hidden
    #expect(!spec.hiddenIndexes.contains(0), "Ordered marker digit should not be hidden")
    #expect(!spec.hiddenIndexes.contains(1), "Ordered marker dot should not be hidden")
    #expect(!spec.hiddenIndexes.contains(2), "Ordered marker space should not be hidden")
  }

  @Test("Ordered list item has no bulletIndexes or hiddenIndexes when cursor inside")
  func noBulletOrHiddenWhenCursorInside() {
    let text = "1. Hello"
    let cursorRange = NSRange(location: 5, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.bulletIndexes.isEmpty, "No bullets for ordered list items")
    #expect(spec.hiddenIndexes.isEmpty, "No hidden chars for ordered list items")
  }

  @Test("Ordered list has no temporary attributes (no delimiter dimming)")
  func noTemporaryAttributesForOrderedList() {
    let text = "1. Hello"
    let cursorRange = NSRange(location: 5, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Ordered list items have no delimiters, so no temp attrs from the list itself
    #expect(spec.temporaryAttributes.isEmpty, "No temp attrs for ordered list items")
  }

  // MARK: - Cursor inside vs outside looks the same

  @Test("Ordered list looks the same whether cursor is inside or outside")
  func cursorInsideOutsideSameForOrderedList() {
    let text = "1. Hello\n\nBody"
    let insideCursor = NSRange(location: 5, length: 0)
    let outsideCursor = NSRange(location: 12, length: 0)

    let insideSpec = MarkdownRenderer.render(text: text, cursorRange: insideCursor)
    let outsideSpec = MarkdownRenderer.render(text: text, cursorRange: outsideCursor)

    // Both should have no bullets and no hidden indexes for the ordered list
    #expect(insideSpec.bulletIndexes.isEmpty, "No bullets when cursor inside")
    #expect(outsideSpec.bulletIndexes.isEmpty, "No bullets when cursor outside")
    #expect(insideSpec.hiddenIndexes.isEmpty, "No hidden when cursor inside")
    #expect(outsideSpec.hiddenIndexes.isEmpty, "No hidden when cursor outside")
  }

  // MARK: - Indentation attributes

  @Test("Ordered list item has paragraph indentation attributes")
  func orderedListItemHasIndentation() {
    let text = "1. Hello\n\nBody"
    let cursorRange = NSRange(location: 12, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have styled ranges for the ordered list item
    let listStyled = spec.styledRanges.first { styled in
      styled.range.location == 0
    }
    #expect(listStyled != nil, "Ordered list item should have styled range")

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

  // MARK: - Multiple ordered list items

  @Test("Multiple ordered list items all have no bullet or hidden treatment")
  func multipleOrderedListItems() {
    let text = "1. Item 1\n2. Item 2\n3. Item 3\n\nBody"
    // Cursor in body
    let cursorRange = NSRange(location: 33, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // No bullets anywhere
    #expect(spec.bulletIndexes.isEmpty, "No bullets for ordered list items")
    // No hidden chars
    #expect(spec.hiddenIndexes.isEmpty, "No hidden chars for ordered list items")
  }

  // MARK: - Mixed ordered and unordered

  @Test("Mixed ordered and unordered: unordered gets bullets, ordered does not")
  func mixedOrderedUnordered() {
    let text = "- Unordered\n1. Ordered\n\nBody"
    // Cursor in body
    let cursorRange = NSRange(location: 26, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Unordered item at position 0 should have bullet
    #expect(spec.bulletIndexes.contains(0), "Unordered marker should be bullet")
    // Unordered space at position 1 should be visible (spacing)
    #expect(!spec.hiddenIndexes.contains(1), "Unordered space should be visible")

    // Ordered item at position 12 should NOT have bullet or be hidden
    #expect(!spec.bulletIndexes.contains(12), "Ordered '1' should not be bullet")
    #expect(!spec.hiddenIndexes.contains(12), "Ordered '1' should not be hidden")
    #expect(!spec.hiddenIndexes.contains(13), "Ordered '.' should not be hidden")
    #expect(!spec.hiddenIndexes.contains(14), "Ordered ' ' should not be hidden")
  }

  // MARK: - Ordered list with inline formatting

  @Test("Bold inside ordered list item works correctly")
  func boldInsideOrderedListItem() {
    let text = "1. **bold** text\n\nBody"
    let cursorRange = NSRange(location: 20, length: 0)  // in body
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Ordered list marker should NOT be bullet or hidden
    #expect(!spec.bulletIndexes.contains(0), "Ordered marker should not be bullet")
    #expect(!spec.hiddenIndexes.contains(0), "Ordered '1' should not be hidden")

    // Bold delimiters should be hidden (cursor is outside bold)
    #expect(spec.hiddenIndexes.contains(3), "Opening ** first char should be hidden")
    #expect(spec.hiddenIndexes.contains(4), "Opening ** second char should be hidden")

    // Bold content should have bold trait
    let boldTrait = spec.fontTraits.first { $0.trait == .boldFontMask }
    #expect(boldTrait != nil, "Should have bold trait inside ordered list item")
  }
}
