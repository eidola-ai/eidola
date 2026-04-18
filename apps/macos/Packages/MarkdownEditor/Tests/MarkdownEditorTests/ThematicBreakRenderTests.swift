import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Thematic Break Render Tests")
@MainActor
struct ThematicBreakRenderTests {

  // MARK: - Attributes when cursor outside

  @Test("Thematic break has transparent text when cursor is outside")
  func transparentTextOutside() {
    let text = "Above\n\n---\n\nBelow"
    let cursorRange = NSRange(location: 0, length: 0)  // cursor in "Above"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // The thematic break range should have foregroundColor: .clear
    let hrStyled = spec.styledRanges.first {
      ($0.attributes[.foregroundColor] as? NSColor) == .clear
    }
    #expect(
      hrStyled != nil,
      "Thematic break should have transparent foreground color when cursor outside")
  }

  @Test("Thematic break has thick strikethrough when cursor is outside")
  func strikethroughOutside() {
    let text = "Above\n\n---\n\nBelow"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hrStyled = spec.styledRanges.first {
      ($0.attributes[.strikethroughStyle] as? Int) == NSUnderlineStyle.thick.rawValue
    }
    #expect(
      hrStyled != nil,
      "Thematic break should have thick strikethrough when cursor outside")
  }

  @Test("Thematic break has separator strikethrough color when cursor is outside")
  func strikethroughColorOutside() {
    let text = "Above\n\n---\n\nBelow"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let style = MarkdownStyle.default
    let hrStyled = spec.styledRanges.first {
      ($0.attributes[.strikethroughColor] as? NSColor) == style.thematicBreakColor
    }
    #expect(
      hrStyled != nil,
      "Thematic break should have separator strikethrough color when cursor outside")
  }

  // MARK: - Attributes when cursor inside

  @Test("Thematic break shows dimmed text when cursor is inside")
  func dimmedTextInside() {
    let text = "Above\n\n---\n\nBelow"
    // "---" starts at offset 7
    let cursorRange = NSRange(location: 8, length: 0)  // inside "---"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have temporary attributes for dimmed color
    let dimmed = spec.temporaryAttributes.first {
      ($0.attributes[.foregroundColor] as? NSColor) == MarkdownStyle.default.delimiterColor
    }
    #expect(
      dimmed != nil,
      "Thematic break should have dimmed text when cursor is inside")
  }

  @Test("Thematic break has no strikethrough when cursor is inside")
  func noStrikethroughInside() {
    let text = "Above\n\n---\n\nBelow"
    let cursorRange = NSRange(location: 8, length: 0)  // inside "---"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should NOT have any styled ranges with strikethrough
    let hasStrikethrough = spec.styledRanges.contains {
      ($0.attributes[.strikethroughStyle] as? Int) == NSUnderlineStyle.thick.rawValue
    }
    #expect(
      !hasStrikethrough,
      "Thematic break should NOT have strikethrough when cursor is inside")
  }

  @Test("Thematic break has no transparent text when cursor is inside")
  func noTransparentTextInside() {
    let text = "Above\n\n---\n\nBelow"
    let cursorRange = NSRange(location: 8, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hasTransparent = spec.styledRanges.contains {
      ($0.attributes[.foregroundColor] as? NSColor) == .clear
    }
    #expect(
      !hasTransparent,
      "Thematic break should NOT have transparent text when cursor is inside")
  }

  // MARK: - Different marker styles

  @Test("Asterisk thematic break (***) works")
  func asteriskThematicBreak() {
    let text = "Above\n\n***\n\nBelow"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hrStyled = spec.styledRanges.first {
      ($0.attributes[.strikethroughStyle] as? Int) == NSUnderlineStyle.thick.rawValue
    }
    #expect(hrStyled != nil, "*** should produce a thematic break with strikethrough")
  }

  @Test("Underscore thematic break (___) works")
  func underscoreThematicBreak() {
    let text = "Above\n\n___\n\nBelow"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hrStyled = spec.styledRanges.first {
      ($0.attributes[.strikethroughStyle] as? Int) == NSUnderlineStyle.thick.rawValue
    }
    #expect(hrStyled != nil, "___ should produce a thematic break with strikethrough")
  }

  // MARK: - Cursor boundary positions

  @Test("Cursor at start of thematic break reveals raw text")
  func cursorAtStart() {
    let text = "Above\n\n---\n\nBelow"
    // "---" starts at offset 7
    let cursorRange = NSRange(location: 7, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hasTransparent = spec.styledRanges.contains {
      ($0.attributes[.foregroundColor] as? NSColor) == .clear
    }
    #expect(
      !hasTransparent,
      "Cursor at start of thematic break should reveal raw text (no transparent)")
  }

  @Test("Cursor at end of thematic break reveals raw text")
  func cursorAtEnd() {
    let text = "Above\n\n---\n\nBelow"
    // "---" ends at offset 10
    let cursorRange = NSRange(location: 10, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hasTransparent = spec.styledRanges.contains {
      ($0.attributes[.foregroundColor] as? NSColor) == .clear
    }
    #expect(
      !hasTransparent,
      "Cursor at end of thematic break should reveal raw text (no transparent)")
  }

  @Test("Cursor just outside thematic break hides text")
  func cursorJustOutside() {
    let text = "Above\n\n---\n\nBelow"
    // Cursor on blank line before "---" (offset 6)
    let cursorRange = NSRange(location: 6, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hasTransparent = spec.styledRanges.contains {
      ($0.attributes[.foregroundColor] as? NSColor) == .clear
    }
    #expect(
      hasTransparent,
      "Cursor just outside thematic break should hide text (transparent)")
  }

  @Test("Cursor on line below thematic break hides text")
  func cursorBelow() {
    let text = "Above\n\n---\n\nBelow"
    // Cursor in "Below" at offset 14
    let cursorRange = NSRange(location: 14, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let hasTransparent = spec.styledRanges.contains {
      ($0.attributes[.foregroundColor] as? NSColor) == .clear
    }
    #expect(
      hasTransparent,
      "Cursor below thematic break should hide text (transparent)")
  }

  // MARK: - Content not hidden via glyph suppression

  @Test("Thematic break does not use glyph hiding")
  func noGlyphHiding() {
    let text = "Above\n\n---\n\nBelow"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // The thematic break characters should NOT be in hiddenIndexes
    // (we use transparent text + strikethrough, not glyph hiding)
    for i in 7...9 {
      #expect(
        !spec.hiddenIndexes.contains(i),
        "Thematic break char at \(i) should not be in hiddenIndexes")
    }
  }
}
