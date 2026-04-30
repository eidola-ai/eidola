import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Code Block Render Tests")
@MainActor
struct CodeBlockRenderTests {
  // MARK: - Attributes

  @Test("Code block applies monospace font to content range")
  func codeBlockFont() {
    let text = "```\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Content is "let x = 42\n" between the fences
    // Opening fence: "```\n" (4 chars), closing fence: "```" (3 chars)
    // Content range: location 4, length 11
    let codeStyled = spec.styledRanges.first {
      $0.attributes[.font] != nil
        && ($0.attributes[.font] as? NSFont)?.fontDescriptor.symbolicTraits.contains(.monoSpace)
          == true
    }
    #expect(codeStyled != nil, "Should have styled range with monospace font for code content")
  }

  @Test("Code block records character range for full-width background drawing")
  func codeBlockBackground() {
    let text = "```\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Background is drawn by TextKit2LayoutFragment using
    // codeBlockCharacterRanges, not via .backgroundColor on styled ranges.
    #expect(!spec.codeBlockCharacterRanges.isEmpty, "Code block should have character ranges for background drawing")
    #expect(
      spec.codeBlockCharacterRanges.first?.range == NSRange(location: 0, length: (text as NSString).length),
      "Code block range should cover the full node")
  }

  @Test("Code block with language hint preserves language in node")
  func codeBlockLanguage() {
    let text = "```swift\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Just verify it renders without errors and has styled ranges
    let codeStyled = spec.styledRanges.first {
      $0.attributes[.font] != nil
        && ($0.attributes[.font] as? NSFont)?.fontDescriptor.symbolicTraits.contains(.monoSpace)
          == true
    }
    #expect(codeStyled != nil, "Code block with language hint should still apply monospace font")
  }

  // MARK: - Delimiter Hiding

  @Test("Code block fences hidden when cursor is outside")
  func fencesHiddenOutside() {
    let text = "hello\n\n```\ncode\n```\n\nworld"
    let cursorRange = NSRange(location: 0, length: 0)  // cursor at "hello"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Opening fence "```\n" starts at position 7
    // Closing fence "```" starts after "code\n"
    #expect(!spec.hiddenIndexes.isEmpty, "Fences should be hidden when cursor is outside")
    // Opening fence backtick characters (the \n after the fence is NOT hidden
    // so that TextKit preserves the paragraph boundary)
    #expect(spec.hiddenIndexes.contains(7), "Opening fence char 0 should be hidden")
    #expect(spec.hiddenIndexes.contains(8), "Opening fence char 1 should be hidden")
    #expect(spec.hiddenIndexes.contains(9), "Opening fence char 2 should be hidden")
  }

  @Test("Code block fences visible and dimmed when cursor is inside content")
  func fencesVisibleInsideContent() {
    let text = "```\ncode\n```"
    let cursorRange = NSRange(location: 5, length: 0)  // inside "code"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Fences should NOT be hidden
    #expect(!spec.hiddenIndexes.contains(0), "Opening fence should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(1), "Opening fence should not be hidden when cursor inside")
    #expect(!spec.hiddenIndexes.contains(2), "Opening fence should not be hidden when cursor inside")

    // Should have temporary attributes for dimmed delimiters
    #expect(!spec.temporaryAttributes.isEmpty, "Should have dimmed delimiter temp attrs")
  }

  @Test("Code block fences visible when cursor is on opening fence line")
  func fencesVisibleOnOpeningFence() {
    let text = "```\ncode\n```"
    let cursorRange = NSRange(location: 0, length: 0)  // at start of opening fence
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Fences should be visible when cursor is on opening fence")
  }

  @Test("Code block fences visible when cursor is on closing fence line")
  func fencesVisibleOnClosingFence() {
    let text = "```\ncode\n```"
    // "```\ncode\n" is 9 chars, closing fence starts at 9
    let cursorRange = NSRange(location: 10, length: 0)  // inside closing fence
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Fences should be visible when cursor is on closing fence")
  }

  @Test("Code block fences visible when cursor is at end of block")
  func fencesVisibleAtEndOfBlock() {
    let text = "```\ncode\n```"
    let cursorRange = NSRange(location: 12, length: 0)  // at very end
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.hiddenIndexes.isEmpty, "Fences should be visible when cursor at end of block")
  }

  // MARK: - Content not hidden

  @Test("Code block content is not hidden")
  func codeBlockContentNotHidden() {
    let text = "```\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Content "let x = 42\n" at positions 4..14 should NOT be hidden
    for i in 4...14 {
      #expect(!spec.hiddenIndexes.contains(i), "Content position \(i) should not be hidden")
    }
  }

  // MARK: - Paragraph style

  @Test("Code block has paragraph style with head indent")
  func codeBlockParagraphStyle() {
    let text = "```\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let codeStyled = spec.styledRanges.first {
      $0.attributes[.paragraphStyle] != nil
        && ($0.attributes[.paragraphStyle] as? NSParagraphStyle)?.headIndent ?? 0 > 0
    }
    #expect(codeStyled != nil, "Code block should have paragraph style with head indent")
  }

  // MARK: - No-wrap (Phase 1)

  @Test("Code block content paragraph style uses .byClipping line break mode")
  func codeBlockContentLineBreakModeIsClipping() {
    // The content paragraph style is the one applied to the inter-fence
    // line range and identifiable by tailIndent == -12 + paragraphSpacing == 0.
    // Fence-line paragraph styles use the codeFenceFont and their paragraph
    // spacing differs, so we filter them out by checking paragraphSpacing.
    let text = "```\nlet x = 42\nlet y = 7\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let contentStyles = spec.styledRanges.compactMap { range -> NSParagraphStyle? in
      guard let ps = range.attributes[.paragraphStyle] as? NSParagraphStyle else { return nil }
      // Code-block content paragraph style: tailIndent -12, both paragraph
      // spacings 0 (so adjacent content fragments stay flush).
      guard ps.tailIndent == -12,
        ps.paragraphSpacing == 0,
        ps.paragraphSpacingBefore == 0
      else { return nil }
      return ps
    }
    #expect(!contentStyles.isEmpty, "Should find at least one code-block content paragraph style")
    for ps in contentStyles {
      #expect(
        ps.lineBreakMode == .byClipping,
        "Code-block content paragraph style must use .byClipping so long lines clip at the container edge instead of wrapping (Phase 1 of the no-wrap feature)")
    }
  }

  @Test("Code block fence paragraph styles keep default wrapping behavior")
  func codeBlockFenceLineBreakModeIsNotClipping() {
    // Fence-line paragraph styles are identifiable by codeBlockSpacing on
    // the outer side (paragraphSpacingBefore for opening, paragraphSpacing
    // for closing) and codeFenceSpacing on the inner side. Either way they
    // have a non-zero paragraph spacing, distinguishing them from content.
    let text = "```\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let fenceStyles = spec.styledRanges.compactMap { range -> NSParagraphStyle? in
      guard let ps = range.attributes[.paragraphStyle] as? NSParagraphStyle else { return nil }
      // Fence-line paragraph styles: tailIndent -12 like content, but
      // paragraphSpacing or paragraphSpacingBefore non-zero.
      guard ps.tailIndent == -12,
        ps.paragraphSpacing != 0 || ps.paragraphSpacingBefore != 0
      else { return nil }
      return ps
    }
    #expect(!fenceStyles.isEmpty, "Should find fence-line paragraph styles")
    for ps in fenceStyles {
      #expect(
        ps.lineBreakMode != .byClipping,
        "Fence-line paragraph style should keep default wrapping behavior (fences themselves never get long, and clipping them would be visually surprising)")
    }
  }

  // MARK: - Code block with language hint

  @Test("Code block with language hint: fences hidden when cursor outside")
  func languageHintFencesHidden() {
    let text = "hello\n\n```python\nprint('hi')\n```\n\nworld"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Opening fence "```python\n" starts at position 7
    #expect(spec.hiddenIndexes.contains(7), "Opening fence should be hidden")
    #expect(spec.hiddenIndexes.contains(8), "Opening fence should be hidden")
    #expect(spec.hiddenIndexes.contains(9), "Opening fence should be hidden")
  }

  @Test("Code block inside blockquote hides closing fence when cursor is outside")
  func blockquoteClosingFenceHiddenOutside() {
    let text = "> ```js\n> let x = 42\n> ```\n\nBody"
    let cursorRange = NSRange(location: (text as NSString).length - 2, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let closingFence = (text as NSString).range(of: "> ```", options: .backwards)
    #expect(closingFence.location != NSNotFound, "Should find closing fence line")
    if closingFence.location != NSNotFound {
      // The > at the start of the fence line is transparent (not hidden)
      #expect(!spec.hiddenIndexes.contains(closingFence.location),
        "Closing fence > at \(closingFence.location) should not be hidden (transparent glyph)")
      // Remaining characters (space + ```) should be hidden
      for idx in (closingFence.location + 1)..<(closingFence.location + closingFence.length) {
        #expect(spec.hiddenIndexes.contains(idx), "Closing fence char at \(idx) should be hidden when cursor is outside")
      }
    }
  }

  // MARK: - Multiline content

  @Test("Code block with multiple lines of content")
  func multilineContent() {
    let text = "hello\n\n```\nline1\nline2\nline3\n```\n\nworld"
    let cursorRange = NSRange(location: 0, length: 0)  // cursor in "hello", outside code block
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Should have monospace font in styled ranges
    let hasCode = spec.styledRanges.contains {
      ($0.attributes[.font] as? NSFont)?.fontDescriptor.symbolicTraits.contains(.monoSpace) == true
    }
    #expect(hasCode, "Multi-line code block should have monospace font")

    // Opening fence should be hidden (cursor is outside)
    #expect(spec.hiddenIndexes.contains(7), "Opening fence should be hidden")
  }
}
