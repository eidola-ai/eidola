import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Inline Code Render Tests")
@MainActor
struct InlineCodeRenderTests {

  // MARK: - Attributes

  @Test("Inline code applies monospace font to content range")
  func inlineCodeFont() {
    let text = "`code`"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have a styled range for the content "code" (positions 1..4)
    let codeStyled = spec.styledRanges.first {
      $0.range == NSRange(location: 1, length: 4)
    }
    #expect(codeStyled != nil, "Should have styled range for code content")

    if let styled = codeStyled {
      let font = styled.attributes[.font] as? NSFont
      #expect(font != nil, "Code content should have a font")
      #expect(
        font?.fontDescriptor.symbolicTraits.contains(.monoSpace) == true,
        "Code font should be monospace")
    }
  }

  @Test("Inline code applies background color to content range")
  func inlineCodeBackground() {
    let text = "`code`"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let codeStyled = spec.styledRanges.first {
      $0.range == NSRange(location: 1, length: 4)
    }
    #expect(codeStyled != nil, "Should have styled range for code content")

    if let styled = codeStyled {
      let bg = styled.attributes[.backgroundColor]
      #expect(bg != nil, "Code content should have a background color")
    }
  }

  // MARK: - Delimiter Hiding

  @Test("Inline code backtick delimiters hidden when cursor is outside")
  func backtickDelimitersHiddenOutside() {
    let text = "hello `code` world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Backtick at position 6 and 11 should be hidden
    #expect(spec.hiddenIndexes.contains(6), "Opening backtick should be hidden")
    #expect(spec.hiddenIndexes.contains(11), "Closing backtick should be hidden")
  }

  @Test("Inline code backtick delimiters visible and dimmed when cursor is inside")
  func backtickDelimitersVisibleInside() {
    let text = "`code`"
    let cursorRange = NSRange(location: 3, length: 0)  // inside "code"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "Opening backtick should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(5), "Closing backtick should not be hidden when cursor inside")

    // Should have temporary attributes for dimmed delimiters
    #expect(spec.temporaryAttributes.count >= 2, "Should have temp attrs for both delimiters")
  }

  // MARK: - Cursor Boundary

  @Test("Cursor at start of inline code reveals delimiters")
  func cursorAtStartOfInlineCode() {
    let text = "`code`"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at start")
  }

  @Test("Cursor at end of inline code reveals delimiters")
  func cursorAtEndOfInlineCode() {
    let text = "`code`"
    let cursorRange = NSRange(location: 6, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at end")
  }

  @Test("Cursor just outside inline code hides delimiters")
  func cursorOutsideInlineCode() {
    let text = "text `code` more"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.isEmpty, "Delimiters should be hidden when cursor outside")
  }

  // MARK: - Inline code with surrounding text

  @Test("Inline code content is not hidden")
  func inlineCodeContentNotHidden() {
    let text = "hello `code` world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // "code" at positions 7..10 should NOT be hidden
    for i in 7...10 {
      #expect(!spec.hiddenIndexes.contains(i), "Content position \(i) should not be hidden")
    }
  }

  @Test("Inline code inside heading gets both heading and code styling")
  func inlineCodeInsideHeading() {
    let text = "# Heading with `code`"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have heading styled range
    #expect(!spec.styledRanges.isEmpty, "Should have styled ranges")

    // Should have code styled range for content "code" (positions 16..19)
    let codeStyled = spec.styledRanges.contains {
      $0.attributes[.backgroundColor] != nil
    }
    #expect(codeStyled, "Should have code background styling")
  }
}
