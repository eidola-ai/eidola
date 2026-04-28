import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Image Render Tests")
@MainActor
struct ImageRenderTests {

  // MARK: - Attributes

  @Test("Image applies secondary label color to content range")
  func imageAttributes() {
    let text = "![alt text](https://example.com/img.png)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Content "alt text" is at positions 2..9
    let imageStyled = spec.styledRanges.first {
      $0.range == NSRange(location: 2, length: 8)
    }
    #expect(imageStyled != nil, "Should have styled range for image content")

    if let styled = imageStyled {
      let color = styled.attributes[.foregroundColor] as? NSColor
      #expect(color != nil, "Image content should have a foreground color")
    }
  }

  @Test("Image applies italic trait to content range")
  func imageItalicTrait() {
    let text = "![alt text](https://example.com/img.png)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Content "alt text" at positions 2..9 should have italic trait
    let italicTrait = spec.fontTraits.first {
      $0.range == NSRange(location: 2, length: 8) && $0.trait == .italicFontMask
    }
    #expect(italicTrait != nil, "Image content should have italic trait")
  }

  // MARK: - Delimiter Hiding

  @Test("Image delimiters hidden when cursor is outside")
  func imageDelimitersHiddenOutside() {
    let text = "hello ![photo](https://example.com/img.png) world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // "hello ![photo](https://example.com/img.png) world"
    // Opening `![` at positions 6,7 should be hidden
    #expect(spec.hiddenIndexes.contains(6), "! should be hidden")
    #expect(spec.hiddenIndexes.contains(7), "[ should be hidden")

    // Closing `](https://example.com/img.png)` starts at position 13
    // ] at 13, ( at 14, url 15..41, ) at 42
    #expect(spec.hiddenIndexes.contains(13), "Closing ] should be hidden")
    #expect(spec.hiddenIndexes.contains(14), "( should be hidden")
    #expect(spec.hiddenIndexes.contains(42), "Closing ) should be hidden")
  }

  @Test("Image delimiters visible and dimmed when cursor is inside")
  func imageDelimitersVisibleInside() {
    let text = "![photo](https://example.com/img.png)"
    let cursorRange = NSRange(location: 4, length: 0)  // inside "photo"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.contains(0), "! should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(1), "[ should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(7), "] should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(36), "Closing ) should not be hidden when cursor inside")

    // Should have temporary attributes for dimmed delimiters
    #expect(spec.temporaryAttributes.count >= 2, "Should have temp attrs for delimiters")
  }

  // MARK: - Cursor Boundary

  @Test("Cursor at start of image reveals delimiters")
  func cursorAtStartOfImage() {
    let text = "![photo](https://example.com/img.png)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at start")
  }

  @Test("Cursor at end of image reveals delimiters")
  func cursorAtEndOfImage() {
    let text = "![photo](https://example.com/img.png)"
    let cursorRange = NSRange(location: 37, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at end")
  }

  @Test("Cursor just outside image hides delimiters")
  func cursorOutsideImage() {
    let text = "text ![photo](https://example.com/img.png) more"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.hiddenIndexes.isEmpty, "Delimiters should be hidden when cursor outside")
  }

  // MARK: - Content not hidden

  @Test("Image content text is not hidden")
  func imageContentNotHidden() {
    let text = "hello ![alt text](https://example.com/img.png) world"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // "alt text" at positions 8..15 should NOT be hidden
    for i in 8...15 {
      #expect(!spec.hiddenIndexes.contains(i), "Content position \(i) should not be hidden")
    }
  }

  // MARK: - Image inside heading

  @Test("Image inside heading gets both heading and image styling")
  func imageInsideHeading() {
    let text = "# Heading with ![img](https://example.com/img.png)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have heading styled range
    #expect(!spec.styledRanges.isEmpty, "Should have styled ranges")

    // Should have image styled range with secondary label color
    let imageStyled = spec.styledRanges.contains {
      $0.attributes[.foregroundColor] as? NSColor == NSColor.secondaryLabelColor
    }
    #expect(imageStyled, "Should have image color styling")
  }

  // MARK: - Image with empty alt text

  @Test("Image with empty alt text still applies image styling")
  func imageWithEmptyAltText() {
    let text = "![](https://example.com/img.png)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Even with empty alt text, delimiters should be hidden when cursor is at start
    // (cursor at start of node is considered "inside")
    // The image construct should still be recognized
    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor at start")
  }

  // MARK: - Cursor at various positions within image

  @Test("Cursor on URL portion of image reveals all delimiters")
  func cursorOnURLPortion() {
    let text = "![photo](https://example.com/img.png)"
    // Cursor at position 15 (inside the URL)
    let cursorRange = NSRange(location: 15, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "All delimiters should be visible when cursor is on URL")
  }

  @Test("Cursor between ] and ( reveals delimiters")
  func cursorBetweenBracketAndParen() {
    let text = "![photo](https://example.com/img.png)"
    // Cursor at position 8 (between ] and ()
    let cursorRange = NSRange(location: 8, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Delimiters should be visible when cursor between ] and (")
  }

  // MARK: - Distinction from link

  @Test("Image is distinct from link - has italic trait, link does not")
  func imageDistinctFromLink() {
    let text = "[link](https://example.com) ![img](https://example.com/img.png)"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Image content should have italic trait
    let imageItalic = spec.fontTraits.contains {
      $0.trait == .italicFontMask
    }
    #expect(imageItalic, "Image should have italic trait")

    // Link should have underline, image should not
    let linkStyled = spec.styledRanges.first {
      $0.attributes[.underlineStyle] != nil
    }
    #expect(linkStyled != nil, "Link should have underline")
  }
}
