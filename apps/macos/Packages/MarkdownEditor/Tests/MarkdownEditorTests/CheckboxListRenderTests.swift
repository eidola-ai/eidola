import AppKit
import Testing

@testable import MarkdownEditor

/// Functional tests for checkbox list rendering: RenderSpec properties.
@Suite("Checkbox List Render Tests")
@MainActor
struct CheckboxListRenderTests {

  // MARK: - Checkbox glyph substitution when cursor outside

  @Test("Unchecked checkbox shows ballot box glyph when cursor is outside")
  func uncheckedCheckboxGlyphWhenCursorOutside() {
    let text = "- [ ] Task\n\nBody"
    // Cursor in body (position 13)
    let cursorRange = NSRange(location: 13, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Position 0 (the '-') should be in uncheckedCheckboxIndexes
    #expect(
      spec.uncheckedCheckboxIndexes.contains(0),
      "Marker char should be an unchecked checkbox glyph")
    // Positions 1-4 (" [ ]") should be hidden, position 5 (space) visible for spacing
    for i in 1...4 {
      #expect(
        spec.hiddenIndexes.contains(i),
        "Position \(i) should be hidden (part of checkbox prefix)")
    }
    #expect(!spec.hiddenIndexes.contains(5), "Trailing space should be visible for spacing")
    // Content should not be hidden or checkbox
    #expect(!spec.hiddenIndexes.contains(6), "Content should not be hidden")
    #expect(!spec.uncheckedCheckboxIndexes.contains(6), "Content should not be checkbox")
    // Should NOT be in bulletIndexes
    #expect(!spec.bulletIndexes.contains(0), "Should not be a regular bullet")
  }

  @Test("Checked checkbox shows ballot box with X when cursor is outside")
  func checkedCheckboxGlyphWhenCursorOutside() {
    let text = "- [x] Done\n\nBody"
    let cursorRange = NSRange(location: 13, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      spec.checkedCheckboxIndexes.contains(0),
      "Marker char should be a checked checkbox glyph")
    // Positions 1-4 hidden, position 5 (space) visible for spacing
    for i in 1...4 {
      #expect(
        spec.hiddenIndexes.contains(i),
        "Position \(i) should be hidden (part of checkbox prefix)")
    }
    #expect(!spec.hiddenIndexes.contains(5), "Trailing space should be visible")
    #expect(!spec.bulletIndexes.contains(0), "Should not be a regular bullet")
  }

  // MARK: - Delimiter dimmed when cursor inside

  @Test("Checkbox delimiter dimmed when cursor is inside")
  func delimiterDimmedWhenCursorInside() {
    let text = "- [ ] Task"
    let cursorRange = NSRange(location: 8, length: 0)  // inside content
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Marker should NOT be in checkbox indexes or hidden
    #expect(
      !spec.uncheckedCheckboxIndexes.contains(0),
      "Marker should not be checkbox glyph when cursor inside")
    #expect(
      !spec.hiddenIndexes.contains(0),
      "Marker char should not be hidden when cursor inside")
    for i in 1...5 {
      #expect(
        !spec.hiddenIndexes.contains(i),
        "Position \(i) should not be hidden when cursor inside")
    }

    // Should have temporary attributes for dimmed delimiter
    #expect(
      spec.temporaryAttributes.count >= 1,
      "Should have temp attrs for dimmed delimiter")
  }

  @Test("Cursor at start of checkbox item reveals delimiter")
  func cursorAtStartRevealsDelimiter() {
    let text = "- [ ] Task"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.uncheckedCheckboxIndexes.isEmpty, "No checkbox glyphs when cursor at start")
    #expect(spec.hiddenIndexes.isEmpty, "Nothing hidden when cursor at start")
  }

  @Test("Cursor at end of checkbox item reveals delimiter")
  func cursorAtEndRevealsDelimiter() {
    let text = "- [ ] Task"
    let cursorRange = NSRange(location: 10, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      spec.uncheckedCheckboxIndexes.isEmpty,
      "No checkbox glyphs when cursor at end of item")
    #expect(
      spec.hiddenIndexes.isEmpty,
      "Nothing hidden when cursor at end of item")
  }

  // MARK: - Mixed list: bullets and checkboxes

  @Test("Mixed list: regular bullet and checkbox each get correct treatment")
  func mixedListBulletAndCheckbox() {
    let text = "- Regular item\n- [ ] Unchecked\n- [x] Checked\n\nBody"
    let cursorRange = NSRange(location: 48, length: 0)  // in body
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // First item (regular): bullet glyph
    #expect(spec.bulletIndexes.contains(0), "Regular item should be bullet")
    #expect(!spec.uncheckedCheckboxIndexes.contains(0), "Regular item should not be checkbox")

    // Second item (unchecked checkbox at position 15)
    #expect(
      spec.uncheckedCheckboxIndexes.contains(15),
      "Unchecked item marker should be unchecked checkbox glyph")
    #expect(!spec.bulletIndexes.contains(15), "Unchecked item should not be bullet")

    // Third item (checked checkbox at position 31)
    #expect(
      spec.checkedCheckboxIndexes.contains(31),
      "Checked item marker should be checked checkbox glyph")
    #expect(!spec.bulletIndexes.contains(31), "Checked item should not be bullet")
  }

  // MARK: - Indentation attributes

  @Test("Checkbox item has paragraph indentation attributes")
  func checkboxItemHasIndentation() {
    let text = "- [ ] Task\n\nBody"
    let cursorRange = NSRange(location: 13, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let listStyled = spec.styledRanges.first { styled in
      styled.range.location == 0
    }
    #expect(listStyled != nil, "Checkbox item should have styled range")

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

  // MARK: - Cursor on one checkbox item reveals only that item

  @Test("Cursor on one checkbox item reveals only that item's delimiter")
  func cursorOnOneItemRevealsOnlyThat() {
    let text = "- [ ] Task A\n- [x] Task B\n\nBody"
    // Cursor inside first item (position 8)
    let cursorRange = NSRange(location: 8, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // First item should NOT have checkbox glyph (cursor inside)
    #expect(
      !spec.uncheckedCheckboxIndexes.contains(0),
      "First item should not have checkbox glyph when cursor inside")

    // Second item SHOULD have checked checkbox glyph (cursor outside)
    #expect(
      spec.checkedCheckboxIndexes.contains(13),
      "Second item should have checked checkbox glyph")
  }
}
