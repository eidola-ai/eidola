import AppKit
import Markdown
import Testing

@testable import MarkdownEditor

@Suite("Heading Cursor Overlap Tests")
@MainActor
struct HeadingCursorOverlapTests {

  @Test("Cursor at end of heading with trailing space reveals delimiter")
  func cursorAtEndWithTrailingSpace() {
    // "# Hello " — cursor at position 8 (end of text)
    // The parser reports heading range as {0, 7} but the cursor is still
    // on the heading line, so delimiters should be visible.
    let text = "# Hello "
    let cursorRange = NSRange(location: 8, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Delimiters should NOT be hidden (cursor is on the heading line)
    #expect(spec.hiddenIndexes.isEmpty, "Delimiter should be visible when cursor is at end of heading line")
    // Should have a temporary attribute for dimmed delimiter color
    #expect(!spec.temporaryAttributes.isEmpty, "Delimiter should have dimmed color")
  }

  @Test("Cursor on next line after heading hides delimiter")
  func cursorOnNextLineHidesDelimiter() {
    // "# Title\n" — cursor at position 8 (start of line 2)
    let text = "# Title\n"
    let cursorRange = NSRange(location: 8, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Delimiters SHOULD be hidden (cursor is on the next line)
    #expect(!spec.hiddenIndexes.isEmpty, "Delimiter should be hidden when cursor is on next line")
    // No temporary attributes for delimiter dimming
    #expect(spec.temporaryAttributes.isEmpty, "No dimmed delimiter when cursor is outside heading")
  }

  @Test("Cursor inside heading content reveals delimiter")
  func cursorInsideHeadingRevealsDelimiter() {
    let text = "# Hello"
    let cursorRange = NSRange(location: 3, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiter should be visible when cursor is inside heading")
    #expect(!spec.temporaryAttributes.isEmpty, "Delimiter should have dimmed color")
  }

  @Test("Cursor at exact end of heading reveals delimiter")
  func cursorAtExactEndOfHeading() {
    let text = "# Hello"
    let cursorRange = NSRange(location: 7, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiter should be visible when cursor is at end of heading")
    #expect(!spec.temporaryAttributes.isEmpty, "Delimiter should have dimmed color")
  }

  @Test("Cursor at end of heading line before newline reveals delimiter")
  func cursorAtEndOfHeadingBeforeNewline() {
    // Regression: cursor at position 7 in "# Hello\nBody text"
    // is at the end of the heading content, just before the \n.
    // The delimiter should be visible (not hidden).
    let text = "# Hello\nBody text"
    let cursorRange = NSRange(location: 7, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      spec.hiddenIndexes.isEmpty,
      "Heading delimiter should be visible when cursor is at end of heading line before \\n")
    #expect(
      !spec.temporaryAttributes.isEmpty,
      "Heading delimiter should be dimmed when cursor is on the heading line")
  }

  @Test("Cursor at end of heading line before newline with body paragraph")
  func cursorAtEndOfHeadingBeforeNewlineWithBody() {
    // Same bug but with a blank line separator (common markdown pattern)
    let text = "# Hello\n\nBody text"
    let cursorRange = NSRange(location: 7, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      spec.hiddenIndexes.isEmpty,
      "Heading delimiter should be visible at end of heading before \\n even with blank line after")
    #expect(!spec.temporaryAttributes.isEmpty)
  }

  @Test("Cursor on line after heading hides delimiter (not confused with end-of-heading)")
  func cursorOnLineAfterHeadingStillHides() {
    // Ensure the fix for end-of-heading doesn't break the next-line case
    let text = "# Hello\nBody text"
    let cursorRange = NSRange(location: 8, length: 0)  // start of "Body text"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      !spec.hiddenIndexes.isEmpty,
      "Heading delimiter should be hidden when cursor is on the next line")
    #expect(
      spec.temporaryAttributes.isEmpty,
      "No dimmed attributes when cursor is outside heading")
  }

  @Test("Delimiter gets heading font, not body font")
  func delimiterGetsHeadingFont() {
    let text = "# Hello"
    let style = MarkdownStyle.default
    let cursorRange = NSRange(location: 3, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange, style: style)

    // The styled ranges should cover the full line including the delimiter
    #expect(!spec.styledRanges.isEmpty)

    // Find the styled range that covers the delimiter position (0)
    let delimiterCovered = spec.styledRanges.contains { styled in
      styled.range.location <= 0 && styled.range.location + styled.range.length > 0
    }
    #expect(delimiterCovered, "Delimiter position should be covered by heading styled range")

    // Verify the styled range has the heading font
    if let styledRange = spec.styledRanges.first(where: { $0.range.location == 0 }) {
      let font = styledRange.attributes[.font] as? NSFont
      #expect(font != nil, "Styled range should have a font")
      let headingFont = style.headingFont(level: 1)
      #expect(font?.pointSize == headingFont.pointSize, "Delimiter should use heading font size")
    }
  }
}
