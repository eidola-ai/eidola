import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Link Render Tests")
@MainActor
struct LinkRenderTests {

  // MARK: - Attributes

  @Test("Link applies blue color and underline to content range")
  func linkAttributes() {
    let text = "[click here](https://example.com)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Content "click here" is at positions 1..10
    let linkStyled = spec.styledRanges.first {
      $0.range == NSRange(location: 1, length: 10)
    }
    #expect(linkStyled != nil, "Should have styled range for link content")

    if let styled = linkStyled {
      let color = styled.attributes[.foregroundColor] as? NSColor
      #expect(color != nil, "Link content should have a foreground color")

      let underline = styled.attributes[.underlineStyle] as? Int
      #expect(underline != nil, "Link content should have underline style")
      #expect(underline == NSUnderlineStyle.single.rawValue, "Underline should be single")
    }
  }

  @Test("Link applies .link attribute with valid URL")
  func linkURLAttribute() {
    let text = "[click](https://example.com)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Content "click" at positions 1..5
    let linkStyled = spec.styledRanges.first {
      $0.range == NSRange(location: 1, length: 5)
    }
    #expect(linkStyled != nil, "Should have styled range for link content")

    if let styled = linkStyled {
      let url = styled.attributes[.link] as? URL
      #expect(url != nil, "Link should have .link URL attribute")
      #expect(url?.absoluteString == "https://example.com", "URL should match destination")
    }
  }

  // MARK: - Delimiter Hiding

  @Test("Link delimiters hidden when cursor is outside")
  func linkDelimitersHiddenOutside() {
    let text = "hello [link](https://example.com) world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Opening `[` at position 6 should be hidden
    #expect(spec.hiddenIndexes.contains(6), "Opening [ should be hidden")

    // Closing `](https://example.com)` starts at position 11
    // `]` at position 11, `(` at 12, url..., `)` at 32
    #expect(spec.hiddenIndexes.contains(11), "Closing ] should be hidden")
    #expect(spec.hiddenIndexes.contains(12), "( should be hidden")
    #expect(spec.hiddenIndexes.contains(32), "Closing ) should be hidden")
  }

  @Test("Link delimiters visible and dimmed when cursor is inside")
  func linkDelimitersVisibleInside() {
    let text = "[link](https://example.com)"
    let cursorRange = NSRange(location: 3, length: 0)  // inside "link"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "Opening [ should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(5), "] should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(26), "Closing ) should not be hidden when cursor inside")

    // Should have temporary attributes for dimmed delimiters
    #expect(spec.temporaryAttributes.count >= 2, "Should have temp attrs for delimiters")
  }

  // MARK: - Cursor Boundary

  @Test("Cursor at start of link reveals delimiters")
  func cursorAtStartOfLink() {
    let text = "[link](https://example.com)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at start")
  }

  @Test("Cursor at end of link reveals delimiters")
  func cursorAtEndOfLink() {
    let text = "[link](https://example.com)"
    let cursorRange = NSRange(location: 27, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at end")
  }

  @Test("Cursor just outside link hides delimiters")
  func cursorOutsideLink() {
    let text = "text [link](https://example.com) more"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.isEmpty, "Delimiters should be hidden when cursor outside")
  }

  // MARK: - Content not hidden

  @Test("Link content text is not hidden")
  func linkContentNotHidden() {
    let text = "hello [click here](https://example.com) world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // "click here" at positions 7..16 should NOT be hidden
    for i in 7...16 {
      #expect(!spec.hiddenIndexes.contains(i), "Content position \(i) should not be hidden")
    }
  }

  // MARK: - Link inside heading

  @Test("Link inside heading gets both heading and link styling")
  func linkInsideHeading() {
    let text = "# Heading with [link](https://example.com)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have heading styled range
    #expect(!spec.styledRanges.isEmpty, "Should have styled ranges")

    // Should have link styled range with underline
    let linkStyled = spec.styledRanges.contains {
      $0.attributes[.underlineStyle] != nil
    }
    #expect(linkStyled, "Should have link underline styling")
  }

  // MARK: - Link without URL

  @Test("Link with empty destination still applies link styling")
  func linkWithEmptyDestination() {
    let text = "[link]()"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let linkStyled = spec.styledRanges.first {
      $0.attributes[.underlineStyle] != nil
    }
    #expect(linkStyled != nil, "Should style link even with empty destination")
  }

  // MARK: - Cursor at various positions within link

  @Test("Cursor on URL portion of link reveals all delimiters")
  func cursorOnURLPortion() {
    let text = "[link](https://example.com)"
    // Cursor at position 15 (inside the URL)
    let cursorRange = NSRange(location: 15, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "All delimiters should be visible when cursor is on URL")
  }

  @Test("Cursor between ] and ( reveals delimiters")
  func cursorBetweenBracketAndParen() {
    let text = "[link](https://example.com)"
    // Cursor at position 6 (between ] and ()
    let cursorRange = NSRange(location: 6, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor between ] and (")
  }
}
